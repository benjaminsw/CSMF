# =============================================================================
# Version: WP2-WebDiag-v1.7
# Abbr: WEBDIAG
# File: experiments/web_diag.py
# Description: Flask diagnostic web server — per-image cons/img/alive metrics
#              served as a browser dashboard accessible via IP or SSH tunnel.
#              Loads CSMF checkpoint (C → B → A fallback), runs inference on
#              validation images, renders per-image diagnostic grid.
# Dependencies: flask, torch, matplotlib, csmf, configs.mnist_config
# Changelog:
#   v1.7 (2026-04-16): 3x2 compact grid — 6 cards fit single screen; run/stage
#                      as text badges; expert selector in sidebar; no-scroll layout
#   v1.6 (2026-04-16): Add left sidebar with live directory tree — JS-driven
#                      expand/collapse folders, click .pth to load, active ckpt
#                      highlighted; /api/ls JSON endpoint for tree data
#   v1.5 (2026-04-16): Fix strict=False fallback — filter by shape match before
#                      load_state_dict; strict=False raises on shape mismatch too
#   v1.4 (2026-04-16): Peek-first arch inference — torch.load checkpoint before
#                      build_model() to infer hidden_dim; eliminates startup crash
#                      on dim mismatch; _load_ckpt_flexible retained for /browse loads
#   v1.3 (2026-04-16): _load_ckpt_flexible() — strict=False load with arch mismatch
#                      tolerance; infers hidden_dim from checkpoint state_dict;
#                      rebuilds model when dims differ; used by startup + /load_checkpoint
#   v1.2 (2026-04-16): Add /browse route — directory browser rooted at
#                      CSMF project dir; click .pth file to load as new ckpt
#   v1.1 (2026-04-16): Remove y_deg panel (downsampled 14x14 cannot render);
#                      add stage dropdown (A/B/C) with /switch_stage POST route;
#                      fix pagination buttons (plain text, no icon font literals)
#   v1.0 (2026-04-16): Initial — Flask server, per-image metric grid,
#                      base64 image embed, /api/metrics JSON endpoint,
#                      checkpoint C→B→A fallback, pagination (?page=N&per_page=N)
# =============================================================================

import argparse
import base64
import io
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Path setup — run from project root with PYTHONPATH=.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
BROWSE_ROOT = "/home/benjamin/Documents/CSMF"
_g_args = None  # set in main(), used by routes that call _load_ckpt_flexible

logging.basicConfig(
    level=logging.INFO,
    format="[WEBDIAG %(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("web_diag")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Global server state (set once at startup)
_G: Dict[str, Any] = {
    "model":       None,
    "val_loader":  None,
    "fwd_model":   None,
    "stage":       "?",
    "ckpt_path":   "?",
    "run_name":    "?",
    "active_experts": [],
    "lambda_cons":  0.05,
    "lambda_img":   1.0,
    "lambda_alive": 0.1,
    "device":      "cpu",
    "start_time":  time.time(),
}


# ===========================================================================
# Model loading
# ===========================================================================

def _infer_hidden_dim(ckpt_path: str, fallback: int) -> int:
    """
    Peek at checkpoint state_dict to infer hidden_dim from gate.0.weight.
    gate = Linear(hidden_dim, hidden_dim//2) → weight shape (hidden_dim//2, hidden_dim).
    Returns inferred value or fallback on any failure.
    """
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
        sd = payload.get("state_dict", payload)
        w = sd.get("gate.0.weight", None)
        if w is not None:
            inferred = int(w.shape[1])
            logger.info("_infer_hidden_dim: inferred hidden_dim=%d from %s", inferred, ckpt_path)
            return inferred
    except Exception as e:
        logger.warning("_infer_hidden_dim: peek failed (%s), using fallback=%d", e, fallback)
    return fallback


def _load_checkpoint(ckpt_dir: str, active_experts: List[str], args) -> Tuple[Any, str, str]:
    """
    Load CSMF from best available stage checkpoint (C → B → A fallback).
    Peeks at the first available checkpoint to infer hidden_dim before building
    the model — avoids size-mismatch crashes when checkpoint arch differs from config.

    Returns:
        (model, stage_loaded, ckpt_path)

    Raises:
        RuntimeError: if no checkpoint can be loaded.
    """
    try:
        from experiments.train_csmf import build_model
        from configs.mnist_config import HIDDEN_DIM, NUM_LAYERS, LATENT_DIM, ACTIVE_EXPERTS
    except ImportError as e:
        logger.error("Import failed: %s", e)
        raise RuntimeError(f"Import failed: {e}") from e

    _active = active_experts or ACTIVE_EXPERTS

    # Stage priority: C → B → A
    candidates = [
        (os.path.join(ckpt_dir, "csmf_stage_C.pth"), "C"),
        (os.path.join(ckpt_dir, "csmf_stage_B.pth"), "B"),
        (os.path.join(ckpt_dir, "csmf_stage_A.pth"), "A"),
    ]
    if getattr(args, "ckpt_C", None):
        candidates.insert(0, (args.ckpt_C, "C"))
    if getattr(args, "ckpt_B", None):
        candidates.insert(1, (args.ckpt_B, "B"))
    if getattr(args, "ckpt_A", None):
        candidates.append((args.ckpt_A, "A"))

    # Peek at the first available checkpoint to infer hidden_dim
    first_ckpt = next((p for p, _ in candidates if os.path.isfile(p)), None)
    if first_ckpt is None:
        raise RuntimeError(
            f"No valid checkpoint found in {ckpt_dir}. "
            "Ensure at least csmf_stage_A.pth exists."
        )

    hidden_dim = _infer_hidden_dim(first_ckpt, fallback=HIDDEN_DIM)
    logger.info("Building model | experts=%s | hidden_dim=%d", _active, hidden_dim)

    model = build_model(
        active_experts=_active,
        hidden_dim=hidden_dim,
        num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM,
        logger=logger,
        args=args,
    )

    for ckpt_path, stage in candidates:
        if os.path.isfile(ckpt_path):
            # If this candidate has a different hidden_dim, infer and rebuild
            h = _infer_hidden_dim(ckpt_path, fallback=hidden_dim)
            if h != hidden_dim:
                logger.info(
                    "Candidate %s has hidden_dim=%d (current=%d) — rebuilding.",
                    ckpt_path, h, hidden_dim
                )
                model = build_model(
                    active_experts=_active,
                    hidden_dim=h,
                    num_layers=NUM_LAYERS,
                    latent_dim=LATENT_DIM,
                    logger=logger,
                    args=args,
                )
                hidden_dim = h

            try:
                meta = _load_ckpt_flexible(model, ckpt_path, _active, args)
                logger.info(
                    "Loaded Stage %s checkpoint: %s | epoch=%s | loss=%.6f",
                    stage, ckpt_path,
                    meta.get("epoch", "?"), meta.get("loss", float("nan")),
                )
                return model, stage, ckpt_path
            except Exception as exc:
                logger.error("Failed to load %s: %s — trying next.", ckpt_path, exc)

    raise RuntimeError(
        f"All checkpoint candidates failed in {ckpt_dir}."
    )


def _load_ckpt_flexible(model, ckpt_path: str, active_experts: list, args) -> dict:
    """
    Load a checkpoint with architecture mismatch tolerance.

    Strategy:
      1. Try strict load via model.load_checkpoint() — fastest path.
      2. On size/key mismatch: infer hidden_dim from checkpoint state_dict,
         rebuild model with that dim, retry strict load.
      3. If rebuild also fails: fall back to strict=False and log mismatched keys.

    Returns:
        meta dict (stage, epoch, loss, ...)
    """
    # ── Attempt 1: strict load ────────────────────────────────────────────────
    try:
        return model.load_checkpoint(ckpt_path)
    except Exception as e1:
        mismatch_msg = str(e1)
        if "size mismatch" not in mismatch_msg and "Missing key" not in mismatch_msg:
            raise  # unexpected error — propagate

        logger.warning(
            "_load_ckpt_flexible: strict load failed (%s) — attempting arch inference.",
            ckpt_path
        )

    # ── Attempt 2: infer hidden_dim and rebuild model ─────────────────────────
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
        sd = payload.get("state_dict", payload)

        # Infer hidden_dim from gate.0.weight (first linear layer: hidden_dim → hidden_dim//2)
        gate_w = sd.get("gate.0.weight", None)
        if gate_w is not None:
            inferred_hidden = gate_w.shape[1]  # input features = hidden_dim
        else:
            inferred_hidden = None

        if inferred_hidden is not None:
            from configs.mnist_config import NUM_LAYERS, LATENT_DIM
            from experiments.train_csmf import build_model
            logger.info(
                "_load_ckpt_flexible: inferred hidden_dim=%d from checkpoint (current=%d) — rebuilding model.",
                inferred_hidden, model.experts[0].dim if hasattr(model.experts[0], 'dim') else -1
            )
            new_model = build_model(
                active_experts=active_experts,
                hidden_dim=inferred_hidden,
                num_layers=NUM_LAYERS,
                latent_dim=LATENT_DIM,
                logger=logger,
                args=args,
            )
            try:
                meta2 = new_model.load_checkpoint(ckpt_path)
                # Replace global model in-place
                model.__class__ = new_model.__class__
                model.__dict__.update(new_model.__dict__)
                logger.info(
                    "_load_ckpt_flexible: rebuilt model with hidden_dim=%d and loaded successfully.",
                    inferred_hidden
                )
                return meta2
            except Exception as e2:
                logger.warning(
                    "_load_ckpt_flexible: rebuilt model still failed (%s) — falling back to strict=False.",
                    e2
                )
    except Exception as e_infer:
        logger.warning("_load_ckpt_flexible: arch inference failed: %s", e_infer)

    # ── Attempt 3: shape-filtered load ──────────────────────────────────────
    # strict=False still raises on shape mismatch — must filter manually first.
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
        sd = payload.get("state_dict", payload)
        current_sd = model.state_dict()

        matched   = {k: v for k, v in sd.items()
                     if k in current_sd and v.shape == current_sd[k].shape}
        skipped   = [k for k, v in sd.items()
                     if k in current_sd and v.shape != current_sd[k].shape]
        missing   = [k for k in current_sd if k not in sd]

        model.load_state_dict(matched, strict=False)

        logger.warning(
            "_load_ckpt_flexible: shape-filtered load | matched=%d / total=%d | "
            "shape-skipped=%d | missing=%d",
            len(matched), len(sd), len(skipped), len(missing)
        )
        if skipped:
            logger.warning(
                "_load_ckpt_flexible: shape-skipped keys (first 5): %s", skipped[:5]
            )

        meta = {k: v for k, v in payload.items() if k != "state_dict"}
        logger.info(
            "_load_ckpt_flexible: partial load complete | stage=%s | epoch=%s",
            meta.get("stage", "?"), meta.get("epoch", "?")
        )
        return meta
    except Exception as e3:
        logger.error("_load_ckpt_flexible: all load strategies failed: %s", e3)
        raise RuntimeError(f"Cannot load checkpoint {ckpt_path}: {e3}") from e3


def _build_forward_model():
    """Construct SRForwardModel matching training config."""
    try:
        from csmf.physics.forward_models import SRForwardModel
        from configs.mnist_config import BLUR_SIGMA, DOWNSAMPLE_FACTOR
        fwd = SRForwardModel(blur_sigma=BLUR_SIGMA, downsample_factor=DOWNSAMPLE_FACTOR)
        logger.info("SRForwardModel built | blur_sigma=%.2f | downsample=%d",
                    BLUR_SIGMA, DOWNSAMPLE_FACTOR)
        return fwd
    except Exception as e:
        logger.error("Failed to build SRForwardModel: %s", e)
        raise


def _build_val_loader(preprocessed_dir: str, batch_size: int):
    """
    Create validation DataLoader from preprocessed data.

    Mirrors the exact call in train_csmf.py main():
        create_precomputed_dataloaders(preprocessed_dir, batch_size,
                                       config_params, worker_init_fn, generator)
    Returns train_loader, val_loader, test_loader — we use index [1].
    """
    try:
        from scripts.preprocess_mnist import create_precomputed_dataloaders
        from configs.mnist_config import (
            BLUR_KERNEL, BLUR_SIGMA, DOWNSAMPLE_FACTOR, NOISE_SIGMA,
            VAL_SPLIT, SEED, make_worker_init_fn,
        )

        config_params = {
            "blur_kernel_size":  BLUR_KERNEL,
            "blur_sigma":        BLUR_SIGMA,
            "downsample_factor": DOWNSAMPLE_FACTOR,
            "noise_std":         NOISE_SIGMA,
            "normalize":         "[0,1]",
            "val_split":         VAL_SPLIT,
            "seed":              SEED,
        }
        _g = torch.Generator()
        _g.manual_seed(SEED)

        _, val_loader, _ = create_precomputed_dataloaders(
            preprocessed_dir=preprocessed_dir,
            batch_size=batch_size,
            config_params=config_params,
            worker_init_fn=make_worker_init_fn(SEED),
            generator=_g,
        )
        logger.info("Val loader created | batch_size=%d | batches=%d",
                    batch_size, len(val_loader))
        return val_loader
    except Exception as e:
        logger.error("Failed to build val loader: %s", e)
        raise


# ===========================================================================
# Inference helpers
# ===========================================================================

def _tensor_to_b64_png(t: torch.Tensor) -> str:
    """
    Convert a (784,) or (28,28) float tensor in [0,1] to base64-encoded PNG.

    Returns empty string on failure (non-fatal).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arr = t.detach().cpu().float()
        if arr.dim() == 1:
            arr = arr.reshape(28, 28)
        arr = arr.clamp(0, 1).numpy()

        fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        fig.tight_layout(pad=0)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=60, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
    except Exception as exc:
        logger.error("_tensor_to_b64_png failed: %s", exc)
        return ""


def _neff(w: torch.Tensor) -> float:
    """
    Effective number of experts.
    w: (K,) gate weights (softmax output, sum=1).
    Neff = exp(H(w)) = exp(-Σ w_k log(w_k))
    """
    w_clamped = w.clamp(min=1e-8)
    H = -(w_clamped * w_clamped.log()).sum()
    return H.exp().item()


def _status_label(cons: float, neff: float, neff_threshold: float = 1.5) -> str:
    """Classify diagnostic status for colour-coding."""
    if cons > 0.10 or neff < 1.1:
        return "CRITICAL"
    if neff < neff_threshold or cons > 0.05:
        return "DRIFT"
    return "STABLE"


@torch.no_grad()
def run_inference(n_images: int, page: int = 1, per_page: int = 6, expert_idx: int = -1) -> Dict[str, Any]:
    """
    expert_idx: -1 = mixture (default), 0..K-1 = individual expert
    """
    # (docstring replaces original, kept inline)
    """
    Run inference on validation images and return per-image diagnostics.

    Args:
        n_images:  total number of images to evaluate
        page:      pagination page (1-indexed)
        per_page:  images per page

    Returns:
        dict with keys: results (list), total, page, per_page, neff_history
    """
    model     = _G["model"]
    val_loader = _G["val_loader"]
    fwd_model  = _G["fwd_model"]
    device     = _G["device"]
    lam_cons   = _G["lambda_cons"]
    lam_img    = _G["lambda_img"]
    lam_alive  = _G["lambda_alive"]

    if model is None or val_loader is None:
        logger.error("run_inference called before model/val_loader initialised")
        raise RuntimeError("Server not fully initialised")

    model.eval()
    fwd_model.eval() if hasattr(fwd_model, "eval") else None

    all_results: List[Dict] = []
    collected = 0

    for batch_idx, batch in enumerate(val_loader):
        if collected >= n_images:
            break

        try:
            x_clean, y_deg = batch[0].to(device), batch[1].to(device)
        except (TypeError, IndexError) as e:
            logger.error("Unexpected batch format at batch %d: %s", batch_idx, e)
            continue

        B = x_clean.shape[0]

        try:
            # w: (B, K)  x_hats: (B, K, d)
            w, x_hats = model.sample_all_experts(y_deg)
            if expert_idx >= 0 and expert_idx < x_hats.shape[1]:
                x_hat_mix = x_hats[:, expert_idx, :]             # (B, d) single expert
            else:
                x_hat_mix = (w.unsqueeze(-1) * x_hats).sum(dim=1)  # (B, d) mixture
        except Exception as exc:
            logger.error("sample_all_experts failed at batch %d: %s", batch_idx, exc)
            continue

        x_clean_flat = x_clean.flatten(1)  # (B, 784)

        for i in range(B):
            if collected >= n_images:
                break

            try:
                # ----- per-image metrics -----
                x_hat_i = x_hat_mix[i]     # (784,)
                x_i     = x_clean_flat[i]  # (784,)
                y_i     = y_deg[i]         # (d',)
                w_i     = w[i]             # (K,)

                # Consistency: λ_cons · ‖A(x̂_mix) − y‖²
                x_hat_img = x_hat_i.reshape(1, 1, 28, 28).clamp(0, 1)
                Ax_hat    = fwd_model(x_hat_img).flatten()
                # y may be downsampled — match shape
                y_flat    = y_i.flatten()
                min_len   = min(Ax_hat.shape[0], y_flat.shape[0])
                cons_raw  = torch.mean((Ax_hat[:min_len] - y_flat[:min_len]) ** 2)
                cons_loss = lam_cons * cons_raw.item()

                # Image loss: λ_img · ‖x̂_mix − x‖²
                img_raw  = torch.mean((x_hat_i - x_i) ** 2)
                img_loss = lam_img * img_raw.item()

                # Neff + alive penalty
                neff_i        = _neff(w_i)
                alive_raw     = max(0.0, 1.5 - neff_i)
                alive_penalty = lam_alive * alive_raw

                # Status label
                status = _status_label(cons_raw.item(), neff_i)

                # Per-expert weights as % for display
                w_pct = {
                    f"w_{k}": round(w_i[k].item() * 100, 1)
                    for k in range(w_i.shape[0])
                }

                # Images → base64
                x_hat_b64  = _tensor_to_b64_png(x_hat_i)
                x_b64      = _tensor_to_b64_png(x_i)

                expert_names = _G.get("active_experts", [])
                if expert_idx >= 0 and expert_idx < len(expert_names):
                    x_hat_label = expert_names[expert_idx]
                else:
                    x_hat_label = "x̂_mix"

                all_results.append({
                    "id":            f"DIAG_{collected + 1:03d}",
                    "x_hat_b64":     x_hat_b64,
                    "x_b64":         x_b64,
                    "x_hat_label":   x_hat_label,
                    "cons_loss":     round(cons_loss, 6),
                    "img_loss":      round(img_loss, 6),
                    "alive_penalty": round(alive_penalty, 6),
                    "neff":          round(neff_i, 3),
                    "status":        status,
                    "gate_weights":  w_pct,
                })
                collected += 1

            except Exception as exc:
                logger.error("Per-image inference failed (batch=%d, i=%d): %s",
                             batch_idx, i, exc)
                all_results.append({
                    "id":     f"DIAG_{collected + 1:03d}_ERR",
                    "error":  str(exc),
                    "status": "ERROR",
                })
                collected += 1

    total = len(all_results)
    start = (page - 1) * per_page
    end   = start + per_page
    page_results = all_results[start:end]

    return {
        "results":   page_results,
        "total":     total,
        "page":      page,
        "per_page":  per_page,
        "n_pages":   math.ceil(total / per_page),
    }


# ===========================================================================
# HTML Template (matches provided design)
# ===========================================================================


_BROWSE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Browse Checkpoints — CSMF</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.tailwindcss.com"></script>
<style>body { font-family: "Inter", sans-serif; background:#f1f4f6; color:#2b3437; }</style>
</head>
<body class="min-h-screen p-8">
  <div class="max-w-3xl mx-auto">

    <!-- Header -->
    <div class="flex items-center justify-between mb-6">
      <div>
        <h1 class="text-xl font-bold font-['Space_Grotesk']">Browse Checkpoints</h1>
        <p class="text-xs text-gray-500 font-mono mt-1 break-all">{{ current_path }}</p>
      </div>
      <a href="/" class="px-4 py-2 bg-white border border-gray-200 rounded text-xs font-bold uppercase tracking-widest hover:bg-gray-50">← Dashboard</a>
    </div>

    <!-- Breadcrumb -->
    <div class="flex items-center gap-1 mb-4 flex-wrap text-xs font-mono text-gray-500">
      {% for crumb in breadcrumbs %}
        <a href="/browse?path={{ crumb.path }}" class="hover:text-blue-600 hover:underline">{{ crumb.name }}</a>
        {% if not loop.last %}<span>/</span>{% endif %}
      {% endfor %}
    </div>

    <!-- Directory listing -->
    <div class="bg-white rounded-xl border border-gray-200 divide-y divide-gray-100 overflow-hidden">

      <!-- Up one level -->
      {% if parent_path %}
      <a href="/browse?path={{ parent_path }}" class="flex items-center gap-3 px-5 py-3 hover:bg-gray-50 transition-colors">
        <span class="text-gray-400 text-lg">↑</span>
        <span class="font-mono text-sm text-gray-500">..</span>
      </a>
      {% endif %}

      <!-- Dirs first -->
      {% for item in dirs %}
      <a href="/browse?path={{ item.path }}" class="flex items-center gap-3 px-5 py-3 hover:bg-gray-50 transition-colors">
        <span class="text-yellow-500">📁</span>
        <span class="font-mono text-sm text-on-surface">{{ item.name }}/</span>
      </a>
      {% endfor %}

      <!-- .pth files -->
      {% for item in pth_files %}
      <form action="/load_checkpoint" method="POST"
            class="flex items-center justify-between px-5 py-3 hover:bg-blue-50 transition-colors cursor-pointer"
            onsubmit="return confirm('Load {{ item.name }}?')">
        <input type="hidden" name="ckpt_path" value="{{ item.path }}"/>
        <div class="flex items-center gap-3">
          <span class="text-blue-500">⚙</span>
          <span class="font-mono text-sm text-blue-700 font-medium">{{ item.name }}</span>
          <span class="text-[10px] text-gray-400">{{ item.size }}</span>
        </div>
        <button type="submit" class="px-3 py-1 bg-blue-600 text-white text-xs font-bold rounded hover:bg-blue-700 transition-colors">Load</button>
      </form>
      {% endfor %}

      {% if not dirs and not pth_files %}
      <div class="px-5 py-8 text-center text-sm text-gray-400 font-mono">No subdirectories or .pth files here.</div>
      {% endif %}
    </div>

  </div>
</body>
</html>
"""

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
{% if auto_refresh %}<meta http-equiv="refresh" content="{{ refresh_secs }}">{% endif %}
<title>CSMF Diagnostics</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700;900&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<style>
  body { font-family:'Inter',sans-serif; background:#f8f9fa; color:#2b3437; overflow:hidden; }
  .font-mono { font-family:'JetBrains Mono',monospace; }
  .status-STABLE   { background:#dcfce7; color:#166534; }
  .status-DRIFT    { background:#fef9c3; color:#854d0e; }
  .status-CRITICAL { background:#fee2e2; color:#991b1b; border-left:3px solid #fe4e49; }
  .status-ERROR    { background:#f3e8ff; color:#6b21a8; }
  .metric-bad  { color:#fe4e49; font-weight:700; }
  .metric-warn { color:#d97706; }
  .metric-ok   { color:#16a34a; font-weight:700; }
  .tree-item:hover { background:rgba(0,0,0,0.04); }
  .active-ckpt { background:#eff6ff; }
  ::-webkit-scrollbar { width:3px; } ::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:2px; }
</style>
</head>
<body class="overflow-hidden">
<div class="flex h-screen overflow-hidden">

<!-- Sidebar -->
<aside class="w-64 flex-shrink-0 bg-slate-50 border-r border-slate-200 flex flex-col h-screen overflow-hidden">
  <div class="px-5 pt-5 pb-3 border-b border-slate-200">
    <div class="font-bold text-slate-900 text-sm" style="font-family:Space Grotesk,sans-serif">CSMF Diagnostics</div>
    <div class="text-[9px] text-slate-400 uppercase tracking-widest mt-0.5 font-mono">WP2-WebDiag-v1.7</div>
  </div>
  <div class="px-4 py-2.5 border-b border-slate-100">
    <div class="text-[9px] font-bold uppercase tracking-widest text-slate-400 mb-1.5">Switch Stage</div>
    <form action="/switch_stage" method="POST">
      <select name="stage" onchange="this.form.submit()" class="w-full font-mono text-xs bg-white border border-slate-200 rounded px-2 py-1.5 focus:outline-none cursor-pointer">
        <option value="C" {% if stage == "C" %}selected{% endif %}>Stage C</option>
        <option value="B" {% if stage == "B" %}selected{% endif %}>Stage B</option>
        <option value="A" {% if stage == "A" %}selected{% endif %}>Stage A</option>
      </select>
    </form>
  </div>
  <div class="px-4 py-2.5 border-b border-slate-100">
    <div class="text-[9px] font-bold uppercase tracking-widest text-slate-400 mb-1.5">View Expert</div>
    <div class="flex flex-wrap gap-1">
      <a href="/?page={{ page }}&per_page=6&n={{ total }}&expert=-1"
         class="px-2 py-0.5 rounded text-[9px] font-mono font-bold border transition-colors {% if selected_expert == -1 %}bg-blue-600 text-white border-blue-600{% else %}bg-white border-slate-200 text-slate-600 hover:bg-slate-50{% endif %}">Mix</a>
      {% for i, name in expert_options %}
      <a href="/?page={{ page }}&per_page=6&n={{ total }}&expert={{ i }}"
         class="px-2 py-0.5 rounded text-[9px] font-mono font-bold border transition-colors {% if selected_expert == i %}bg-blue-600 text-white border-blue-600{% else %}bg-white border-slate-200 text-slate-600 hover:bg-slate-50{% endif %}">{{ name }}</a>
      {% endfor %}
    </div>
  </div>
  <div class="flex-1 overflow-y-auto py-2">
    <div class="px-4 mb-1"><span class="text-[9px] font-bold uppercase tracking-widest text-slate-400">Storage Explorer</span></div>
    <div id="tree-root" class="font-mono text-[10px]"><div class="px-4 text-slate-400 text-[9px] py-1">Loading...</div></div>
  </div>
  <div class="px-4 py-2 border-t border-slate-200 bg-slate-100 shrink-0">
    <div class="text-[9px] font-bold uppercase tracking-widest text-slate-400 mb-0.5">Loaded</div>
    <div class="font-mono text-[10px] text-blue-700 truncate" title="{{ ckpt_path }}">{{ ckpt_short }}</div>
  </div>
</aside>

<!-- Main -->
<main class="flex-1 h-screen flex flex-col overflow-hidden bg-slate-50">
  <!-- Topbar -->
  <header class="bg-white/90 backdrop-blur border-b border-slate-200/50 px-6 h-13 flex items-center justify-between shrink-0" style="height:3.25rem">
    <div class="flex items-center gap-5">
      <h1 class="text-base font-black text-slate-900" style="font-family:Space Grotesk,sans-serif">Model Diagnostics</h1>
      <div class="flex items-center gap-3">
        <div class="flex items-center gap-1">
          <span class="text-[9px] font-bold text-slate-400 uppercase">Run:</span>
          <span class="text-[10px] font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded">{{ run_name }}</span>
        </div>
        <div class="flex items-center gap-1">
          <span class="text-[9px] font-bold text-slate-400 uppercase">Stage:</span>
          <span class="text-[10px] font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded">Stage {{ stage }}</span>
        </div>
        <span class="text-[9px] font-mono text-slate-400">λ<sub>c</sub>={{ lam_cons }} λ<sub>i</sub>={{ lam_img }} λ<sub>a</sub>={{ lam_alive }}</span>
      </div>
    </div>
    <div class="flex gap-2">
      <a href="/?page={{ page }}&per_page=6&n={{ total }}&expert={{ selected_expert }}" class="px-3 py-1 bg-slate-100 border border-slate-200 rounded text-[9px] font-bold uppercase tracking-widest hover:bg-slate-200 transition-colors">↻ Refresh</a>
      <a href="/api/metrics?n={{ total }}" target="_blank" class="px-3 py-1 bg-slate-100 border border-slate-200 rounded text-[9px] font-bold uppercase tracking-widest hover:bg-slate-200 transition-colors">JSON</a>
    </div>
  </header>

  <!-- Canvas -->
  <div class="p-4 flex-1 flex flex-col min-h-0 overflow-hidden">

    <!-- 3×2 grid -->
    <div class="grid grid-cols-3 grid-rows-2 gap-3 flex-1 min-h-0">
      {% for r in results %}
      <div class="bg-white p-3 rounded-lg flex flex-col border border-transparent hover:border-slate-200 transition-all shadow-sm min-h-0 {% if r.status == 'CRITICAL' %}border-l-2 border-l-red-400{% endif %}">
        <div class="flex justify-between items-center mb-1.5 shrink-0">
          <span class="text-[11px] font-mono font-bold text-slate-500">{{ r.id }}</span>
          <span class="text-[10px] px-2 py-0.5 rounded font-bold status-{{ r.status }}">{{ r.status }}</span>
        </div>
        {% if r.get('error') %}
        <div class="text-[9px] text-red-700 font-mono p-2 bg-red-50 rounded flex-1">{{ r.error }}</div>
        {% else %}
        <div class="grid grid-cols-2 gap-1.5 flex-1 min-h-0 mb-1.5">
          <div class="flex flex-col min-h-0">
            {% if r.x_hat_b64 %}<img src="{{ r.x_hat_b64 }}" class="w-full flex-1 min-h-0 object-cover rounded contrast-125" style="min-height:0"/>
            {% else %}<div class="flex-1 bg-slate-100 rounded min-h-0"></div>{% endif %}
            <p class="text-[10px] text-center font-mono font-bold text-blue-600 mt-0.5 shrink-0">{{ r.x_hat_label }}</p>
          </div>
          <div class="flex flex-col min-h-0">
            {% if r.x_b64 %}<img src="{{ r.x_b64 }}" class="w-full flex-1 min-h-0 object-cover rounded" style="min-height:0"/>
            {% else %}<div class="flex-1 bg-slate-100 rounded min-h-0"></div>{% endif %}
            <p class="text-[10px] text-center font-mono font-bold text-slate-500 mt-0.5 shrink-0">x_clean</p>
          </div>
        </div>
        <div class="grid grid-cols-4 gap-1 pt-1.5 border-t border-slate-100 shrink-0">
          <div><p class="text-[9px] text-slate-500 uppercase font-bold">Cons</p><p class="text-[11px] font-mono {% if r.cons_loss > 0.05 %}metric-bad{% elif r.cons_loss > 0.02 %}metric-warn{% else %}metric-ok{% endif %}">{{ "%.4f"|format(r.cons_loss) }}</p></div>
          <div><p class="text-[9px] text-slate-500 uppercase font-bold">Img</p><p class="text-[11px] font-mono {% if r.img_loss > 0.1 %}metric-bad{% elif r.img_loss > 0.05 %}metric-warn{% else %}metric-ok{% endif %}">{{ "%.4f"|format(r.img_loss) }}</p></div>
          <div><p class="text-[9px] text-slate-500 uppercase font-bold">Alive</p><p class="text-[11px] font-mono {% if r.alive_penalty > 0.05 %}metric-bad{% elif r.alive_penalty > 0 %}metric-warn{% else %}metric-ok{% endif %}">{{ "%.4f"|format(r.alive_penalty) }}</p></div>
          <div><p class="text-[9px] text-slate-500 uppercase font-bold">Neff</p><p class="text-[11px] font-mono {% if r.neff < 1.1 %}metric-bad{% elif r.neff < 1.5 %}metric-warn{% else %}metric-ok{% endif %}">{{ "%.2f"|format(r.neff) }}</p></div>
        </div>
        <div class="mt-0.5 text-[11px] font-mono text-slate-400 truncate shrink-0">{% for k, v in r.gate_weights.items() %}{{ k }}={{ v }}% {% endfor %}</div>
        {% endif %}
      </div>
      {% endfor %}
      {% if results|length == 0 %}
      <div class="col-span-3 row-span-2 flex items-center justify-center text-slate-400 font-mono text-sm">No results — check server logs.</div>
      {% endif %}
    </div>

    <!-- Bottom bar -->
    <div class="mt-3 flex flex-col gap-2 shrink-0">
      <div class="bg-slate-900 text-white/60 px-4 h-8 flex items-center gap-3 rounded font-mono text-[9px]">
        <span class="text-blue-400 font-bold uppercase">Log:</span>
        <span class="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse shrink-0"></span>
        <span class="flex-1 truncate">Stage {{ stage }} | {{ ckpt_short }} | uptime {{ uptime }}</span>
        <span class="text-white/30 shrink-0">{% if auto_refresh %}auto {{ refresh_secs }}s{% else %}live{% endif %}</span>
      </div>
      <div class="bg-white px-5 py-2 flex items-center justify-between border border-slate-200/50 rounded">
        <div class="flex items-center gap-2">
          <span class="text-[9px] font-mono text-slate-400">Page {{ page }} of {{ n_pages }}</span>
          <span class="h-1 w-1 rounded-full bg-slate-300"></span>
          <span class="text-[9px] font-mono text-slate-400">{{ total }} samples</span>
        </div>
        <div class="flex gap-2">
          <a href="/?page={{ [page-1,1]|max }}&per_page=6&n={{ total }}&expert={{ selected_expert }}"
             class="px-4 py-1.5 bg-slate-100 text-slate-500 text-[9px] font-bold rounded hover:bg-slate-200 transition-colors {% if page <= 1 %}opacity-40 pointer-events-none{% endif %}">← Back</a>
          <a href="/?page={{ [page+1,n_pages]|min }}&per_page=6&n={{ total }}&expert={{ selected_expert }}"
             class="px-4 py-1.5 bg-blue-600 text-white text-[9px] font-bold rounded hover:bg-blue-700 transition-colors {% if page >= n_pages %}opacity-40 pointer-events-none{% endif %}">Next →</a>
        </div>
      </div>
    </div>
  </div>
</main>
</div>

<script>
const ACTIVE_CKPT = {{ ckpt_path | tojson }};
const expanded = new Set();
(function(){
  try {
    const parts = ACTIVE_CKPT.replace(/^\/home\/benjamin\/Documents\/CSMF\/?/, "").split("/");
    let p = "";
    for (let i = 0; i < parts.length - 1; i++) { p = p ? p+"/"+parts[i] : parts[i]; expanded.add(p); }
  } catch(e) {}
})();

async function fetchDir(rel){ const r = await fetch("/api/ls?path="+encodeURIComponent(rel)); return r.json(); }

async function renderTree(container, relPath, depth) {
  container.innerHTML = '<div style="padding-left:'+(depth*10+10)+'px" class="text-slate-400 text-[9px] py-0.5">loading...</div>';
  let data; try { data = await fetchDir(relPath); } catch(e) { container.innerHTML='<div class="px-3 text-red-400 text-[9px]">Error</div>'; return; }
  container.innerHTML = "";
  const indent = depth * 10;
  for (const item of data) {
    const row = document.createElement("div");
    if (item.type === "dir") {
      const isOpen = expanded.has(item.rel);
      row.className = "tree-item cursor-pointer select-none";
      row.style.paddingLeft = (indent+4)+"px";
      row.innerHTML = `<div class="flex items-center gap-1 py-0.5 pr-2"><span class="text-slate-400 text-[9px] w-3 shrink-0">${isOpen?"▾":"▸"}</span><span class="text-[10px]">📁</span><span class="text-[10px] text-slate-700 truncate">${item.name}</span></div>`;
      const cc = document.createElement("div");
      cc.style.display = isOpen ? "block" : "none";
      if (isOpen) renderTree(cc, item.rel, depth+1);
      row.addEventListener("click", async(e)=>{ e.stopPropagation(); const open=cc.style.display!=="none"; if(open){cc.style.display="none";expanded.delete(item.rel);row.querySelector("span").textContent="▸";}else{cc.style.display="block";expanded.add(item.rel);row.querySelector("span").textContent="▾";if(!cc.hasChildNodes())renderTree(cc,item.rel,depth+1);}});
      container.appendChild(row); container.appendChild(cc);
    } else if (item.type === "pth") {
      const isActive = item.abs === ACTIVE_CKPT;
      row.className = "tree-item cursor-pointer select-none" + (isActive?" active-ckpt":"");
      row.style.paddingLeft = (indent+4)+"px";
      row.innerHTML = `<div class="flex items-center gap-1 py-0.5 pr-2"><span class="w-3 shrink-0"></span><span class="text-[9px] ${isActive?"text-blue-700 font-bold":"text-slate-500"} truncate flex-1">${item.name}</span><span class="text-[7px] text-slate-300 shrink-0">${item.size}</span></div>`;
      if (!isActive) row.addEventListener("click",(e)=>{ e.stopPropagation(); if(confirm("Load "+item.name+"?")){ const f=document.createElement("form");f.method="POST";f.action="/load_checkpoint";const inp=document.createElement("input");inp.type="hidden";inp.name="ckpt_path";inp.value=item.abs;f.appendChild(inp);document.body.appendChild(f);f.submit();}});
      container.appendChild(row);
    }
  }
  if (data.length===0){const e=document.createElement("div");e.style.paddingLeft=(indent+14)+"px";e.className="text-[9px] text-slate-300 py-0.5";e.textContent="empty";container.appendChild(e);}
}
renderTree(document.getElementById("tree-root"), "", 0);
</script>
</body>
</html>
"""




# ===========================================================================
# Routes
# ===========================================================================

@app.route("/")
def index():
    """Main diagnostic dashboard."""
    try:
        page     = max(1, int(request.args.get("page",     1)))
        per_page = max(1, int(request.args.get("per_page", 6)))
        n        = max(1, int(request.args.get("n",        24)))
        auto_ref = request.args.get("auto_refresh", "0") == "1"
        ref_secs = max(5, int(request.args.get("refresh_secs", 30)))
        expert   = int(request.args.get("expert", -1))

        data = run_inference(n_images=n, page=page, per_page=per_page, expert_idx=expert)

        uptime_sec = int(time.time() - _G["start_time"])
        h, rem = divmod(uptime_sec, 3600)
        m, s   = divmod(rem, 60)

        ckpt_short = os.path.basename(_G["ckpt_path"])
        experts_str = ", ".join(_G["active_experts"])

        expert_options = list(enumerate(_G["active_experts"]))

        return render_template_string(
            _HTML_TEMPLATE,
            results          = data["results"],
            total            = data["total"],
            page             = data["page"],
            per_page         = data["per_page"],
            n_pages          = data["n_pages"],
            run_name         = _G["run_name"],
            stage            = _G["stage"],
            ckpt_path        = _G["ckpt_path"],
            ckpt_short       = ckpt_short,
            uptime           = f"{h:02d}h {m:02d}m",
            lam_cons         = _G["lambda_cons"],
            lam_img          = _G["lambda_img"],
            lam_alive        = _G["lambda_alive"],
            experts_str      = experts_str,
            auto_refresh     = auto_ref,
            refresh_secs     = ref_secs,
            selected_expert  = expert,
            expert_options   = expert_options,
        )
    except Exception as exc:
        logger.error("index() error: %s", exc, exc_info=True)
        return f"<pre style='color:red'>Server error: {exc}</pre>", 500


@app.route("/api/metrics")
def api_metrics():
    """JSON endpoint — returns per-image metrics without images."""
    try:
        page     = max(1, int(request.args.get("page",     1)))
        per_page = max(1, int(request.args.get("per_page", 50)))
        n        = max(1, int(request.args.get("n",        100)))

        data = run_inference(n_images=n, page=page, per_page=per_page)

        # Strip base64 image blobs — API consumers get metrics only
        for r in data["results"]:
            r.pop("y_b64",     None)
            r.pop("x_hat_b64", None)
            r.pop("x_b64",     None)

        data["meta"] = {
            "run_name":    _G["run_name"],
            "stage":       _G["stage"],
            "ckpt_path":   _G["ckpt_path"],
            "lambda_cons":  _G["lambda_cons"],
            "lambda_img":   _G["lambda_img"],
            "lambda_alive": _G["lambda_alive"],
            "experts":      _G["active_experts"],
            "timestamp":    time.time(),
        }
        return jsonify(data)
    except Exception as exc:
        logger.error("api_metrics() error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ls")
def api_ls():
    """Return directory listing JSON for sidebar tree. Dirs + .pth files only."""
    rel = request.args.get("path", "")
    target = os.path.realpath(os.path.join(BROWSE_ROOT, rel.lstrip("/")))
    if not target.startswith(os.path.realpath(BROWSE_ROOT)):
        return jsonify({"error": "access denied"}), 403
    if not os.path.isdir(target):
        return jsonify({"error": "not a directory"}), 404
    try:
        entries = sorted(os.listdir(target))
    except PermissionError:
        return jsonify({"error": "permission denied"}), 403

    result = []
    for name in entries:
        if name.startswith("."):
            continue
        full = os.path.join(target, name)
        rel_full = os.path.relpath(full, BROWSE_ROOT)
        if os.path.isdir(full):
            result.append({"type": "dir", "name": name, "rel": rel_full})
        elif name.endswith(".pth"):
            size_mb = os.path.getsize(full) / 1e6
            result.append({
                "type": "pth",
                "name": name,
                "rel":  rel_full,
                "abs":  full,
                "size": f"{size_mb:.1f}MB",
            })
    return jsonify(result)


@app.route("/browse")
def browse():
    """Directory browser rooted at BROWSE_ROOT for picking .pth checkpoints."""
    import math as _math

    rel = request.args.get("path", "")
    # Resolve and security-check: must stay within BROWSE_ROOT
    target = os.path.realpath(os.path.join(BROWSE_ROOT, rel.lstrip("/")))
    if not target.startswith(os.path.realpath(BROWSE_ROOT)):
        logger.error("browse: path escape attempt: %s", rel)
        return "Access denied", 403

    if not os.path.isdir(target):
        return "Not a directory", 404

    try:
        entries = sorted(os.listdir(target))
    except PermissionError:
        return "Permission denied", 403

    dirs = []
    pth_files = []
    for name in entries:
        full = os.path.join(target, name)
        rel_full = os.path.relpath(full, BROWSE_ROOT)
        if os.path.isdir(full) and not name.startswith("."):
            dirs.append({"name": name, "path": rel_full})
        elif name.endswith(".pth"):
            size_mb = os.path.getsize(full) / 1e6
            pth_files.append({
                "name": name,
                "path": full,   # absolute path passed to /load_checkpoint
                "size": f"{size_mb:.1f} MB",
            })

    # Breadcrumbs
    parts = os.path.relpath(target, BROWSE_ROOT).split(os.sep)
    breadcrumbs = [{"name": "CSMF", "path": ""}]
    for i, p in enumerate(parts):
        if p and p != ".":
            breadcrumbs.append({"name": p, "path": os.path.join(*parts[:i+1])})

    parent_path = None
    parent = os.path.dirname(target)
    if os.path.realpath(parent).startswith(os.path.realpath(BROWSE_ROOT)):
        parent_path = os.path.relpath(parent, BROWSE_ROOT)

    return render_template_string(
        _BROWSE_TEMPLATE,
        current_path = target,
        breadcrumbs  = breadcrumbs,
        dirs         = dirs,
        pth_files    = pth_files,
        parent_path  = parent_path,
    )


@app.route("/load_checkpoint", methods=["POST"])
def load_checkpoint_route():
    """Load an arbitrary .pth file selected from the browser."""
    from flask import redirect, url_for
    ckpt_path = request.form.get("ckpt_path", "").strip()

    # Security: must be within BROWSE_ROOT
    if not os.path.realpath(ckpt_path).startswith(os.path.realpath(BROWSE_ROOT)):
        logger.error("load_checkpoint: path escape attempt: %s", ckpt_path)
        return "Access denied", 403

    if not os.path.isfile(ckpt_path):
        return f"<pre style='color:red'>File not found: {ckpt_path}</pre>", 404

    try:
        meta = _load_ckpt_flexible(_G["model"], ckpt_path, _G["active_experts"], _g_args)
        _G["model"].eval()
        _G["model"].to(_G["device"])
        _G["stage"]     = meta.get("stage", "?")
        _G["ckpt_path"] = ckpt_path
        logger.info(
            "load_checkpoint: loaded %s | stage=%s | epoch=%s | loss=%.6f",
            ckpt_path, _G["stage"],
            meta.get("epoch", "?"), meta.get("loss", float("nan")),
        )
    except Exception as exc:
        logger.error("load_checkpoint: failed: %s", exc)
        return f"<pre style='color:red'>Load failed: {exc}</pre>", 500

    return redirect(url_for("index"))


@app.route("/switch_stage", methods=["POST"])
def switch_stage():
    """Reload model from a different stage checkpoint (A / B / C)."""
    from flask import redirect, url_for
    requested = request.form.get("stage", "C").upper()
    if requested not in ("A", "B", "C"):
        logger.error("switch_stage: invalid stage '%s'", requested)
        return "Invalid stage", 400

    ckpt_dir = os.path.dirname(_G["ckpt_path"])
    ckpt_map = {
        "A": os.path.join(ckpt_dir, "csmf_stage_A.pth"),
        "B": os.path.join(ckpt_dir, "csmf_stage_B.pth"),
        "C": os.path.join(ckpt_dir, "csmf_stage_C.pth"),
    }
    ckpt_path = ckpt_map[requested]

    if not os.path.isfile(ckpt_path):
        logger.error("switch_stage: checkpoint not found: %s", ckpt_path)
        return f"<pre style='color:red'>Checkpoint not found: {ckpt_path}</pre>", 404

    try:
        meta = _load_ckpt_flexible(_G["model"], ckpt_path, _G["active_experts"], _g_args)
        _G["model"].eval()
        _G["model"].to(_G["device"])
        _G["stage"]     = requested
        _G["ckpt_path"] = ckpt_path
        logger.info("switch_stage: loaded Stage %s | epoch=%s | loss=%.6f",
                    requested, meta.get("epoch","?"), meta.get("loss", float("nan")))
    except Exception as exc:
        logger.error("switch_stage: failed to load %s: %s", ckpt_path, exc)
        return f"<pre style='color:red'>Load failed: {exc}</pre>", 500

    return redirect(url_for("index"))




# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="CSMF Web Diagnostic Server (WEBDIAG-v1.0)")
    p.add_argument("--run-name",         default="no_glow_no_maf_v3")
    p.add_argument("--ckpt-dir",         default="logs/reconstruction_first/checkpoints")
    p.add_argument("--preprocessed-dir", default="./data/preprocessed")
    p.add_argument("--port",             type=int, default=5000)
    p.add_argument("--host",             default="0.0.0.0",
                   help="Use 0.0.0.0 for LAN/SSH-tunnel access")
    p.add_argument("--n-images",         type=int, default=24,
                   help="Total val images to evaluate per page load")
    p.add_argument("--per-page",         type=int, default=6)
    p.add_argument("--batch-size",       type=int, default=16)
    p.add_argument("--lambda-img",       type=float, default=1.0)
    p.add_argument("--lambda-alive",     type=float, default=0.1)
    p.add_argument("--experts",          nargs="+",
                   default=["realnvp", "nice", "nsf", "csf"])
    p.add_argument("--ckpt-A",           default=None)
    p.add_argument("--ckpt-B",           default=None)
    p.add_argument("--ckpt-C",           default="/home/benjamin/Documents/CSMF/logs/reconstruction_first/checkpoints/csmf_stage_C.pth")
    p.add_argument("--nice-scale",       type=float, default=0.05)
    p.add_argument("--device",           default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = _parse_args()

    from configs.mnist_config import LAMBDA_CONS, ACTIVE_EXPERTS

    logger.info("=" * 60)
    logger.info("CSMF Web Diagnostic Server — WP2-WebDiag-v1.0")
    logger.info("Run:    %s", args.run_name)
    logger.info("Ckpt:   %s", args.ckpt_dir)
    logger.info("Device: %s", args.device)
    logger.info("=" * 60)

    # --- Load model ---
    model, stage, ckpt_path = _load_checkpoint(
        ckpt_dir=args.ckpt_dir,
        active_experts=args.experts,
        args=args,
    )
    model.eval()
    model.to(args.device)

    # --- Forward model ---
    fwd_model = _build_forward_model()
    fwd_model.to(args.device)

    # --- Val loader ---
    val_loader = _build_val_loader(
        preprocessed_dir=args.preprocessed_dir,
        batch_size=args.batch_size,
    )

    # --- Store globals ---
    _G.update({
        "model":          model,
        "args":           args,
        "val_loader":     val_loader,
        "fwd_model":      fwd_model,
        "stage":          stage,
        "ckpt_path":      ckpt_path,
        "run_name":       args.run_name,
        "active_experts": args.experts,
        "lambda_cons":    LAMBDA_CONS,
        "lambda_img":     args.lambda_img,
        "lambda_alive":   args.lambda_alive,
        "device":         args.device,
        "start_time":     time.time(),
    })
    global _g_args
    _g_args = args

    logger.info("Server starting at http://%s:%d", args.host, args.port)
    logger.info("Access locally: http://localhost:%d", args.port)
    logger.info("SSH tunnel (from your machine): ssh -L %d:localhost:%d user@<REMOTE_IP>",
                args.port, args.port)

    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
