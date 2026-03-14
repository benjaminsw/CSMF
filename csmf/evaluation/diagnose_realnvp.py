"""
RealNVP Diagnostic Script — Flat NLL Investigation

Version: DIAG-RNVP-v1.1.0
Abbr: DIAG-RNVP
Last Modified: 2026-03-05
Changelog:
  v1.1.0 (2026-03-05): [F] Replaced hook-based check3/checkC with direct manual forward — ScaleBlock
                        calls coupling.forward() directly, bypassing __call__ and hooks. Added empty-
                        result guard to summary logic. Fixed checkB plotting for extreme z values
                        (adaptive binning + percentile-based x-axis limits). Added vacuous PASS
                        detection in summary.
  v1.0.2 (2026-03-01): [B] Handle full CSMF checkpoint — extract expert 0 sub-dict.
  v1.0.1 (2026-03-01): [B] Fixed kwarg data_dir → preprocessed_dir.
  v1.0   (2026-03-01): Initial implementation — 6 diagnostic checks for RealNVP flat-line NLL.

Usage:
    python diagnose_realnvp.py --checkpoint path/to/expert_realnvp.pth --data_dir ./data/preprocessed
    python diagnose_realnvp.py --no-checkpoint  # random init baseline (sanity check the checks)

Outputs:
    results/diag_realnvp/
        check1_logdet_stats.csv
        check2_sample_grid.png
        check2_sample_stats.csv
        check3_coupling_delta.csv
        checkA_logdet_violin.png
        checkB_latent_z_hist.png
        checkC_st_network_stats.csv
        diagnostic_summary.json
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import csv
import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("DIAG-RNVP")

VERSION = "DIAG-RNVP-v1.1.0"
logger.info(f"Starting {VERSION}")

OUTPUT_DIR = "results/diag_realnvp"


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    return OUTPUT_DIR


def load_model_and_data(checkpoint_path, data_dir, device, n_val_batches=5, batch_size=64):
    try:
        from csmf.flows.conditional_realnvp import ConditionalRealNVP
        from scripts.preprocess_mnist import create_precomputed_dataloaders
    except ImportError as e:
        logger.error(f"Import failed: {e}. Ensure project is on PYTHONPATH.")
        raise

    try:
        _, val_loader, _ = create_precomputed_dataloaders(
            preprocessed_dir=data_dir, batch_size=batch_size, num_workers=0
        )
        logger.info(f"Loaded validation data from {data_dir}")
    except Exception as e:
        logger.error(f"Failed to load data from {data_dir}: {e}")
        raise

    model = ConditionalRealNVP(h_dim=64, hidden_dims=[256, 256], debug=False)

    if checkpoint_path is not None:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt

            if any(k.startswith('experts.0.') for k in state.keys()):
                state = {k.replace('experts.0.', ''): v
                         for k, v in state.items() if k.startswith('experts.0.')}
                logger.info("Extracted expert 0 sub-dict from full CSMF checkpoint")

            model.load_state_dict(state)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    else:
        logger.warning("No checkpoint — using random initialization (baseline sanity check)")

    model = model.to(device)
    model.eval()
    return model, val_loader


# ═══════════════════════════════════════════════════════════════════════
# HELPER — Manual forward through scale block couplings
# ═══════════════════════════════════════════════════════════════════════
def _manual_scale_forward(scale_block, z_4d, h):
    """Forward through a scale block's coupling layers, returning per-layer data.

    Calls coupling.forward() directly (same as ScaleBlock does), capturing
    input/output at each layer. This avoids the hook bypass problem.
    """
    B, C, H, W = z_4d.shape
    z_flat = z_4d.reshape(B, -1)
    layer_data = []

    for c_idx, coupling in enumerate(scale_block.coupling_layers):
        z_before = z_flat.clone()
        z_flat, ld = coupling.forward(z_flat, h, reverse=False)

        delta = (z_flat - z_before).norm(dim=1).mean().item()
        in_norm = z_before.norm(dim=1).mean().item()
        rel_delta = delta / (in_norm + 1e-10)
        cos = torch.nn.functional.cosine_similarity(z_before, z_flat, dim=1).mean().item()

        split_dim = coupling.split_dim
        x_A_before = z_before[:, :split_dim]
        x_B_before = z_before[:, split_dim:]
        x_A_after = z_flat[:, :split_dim]
        x_B_after = z_flat[:, split_dim:]

        delta_B = x_B_after - x_B_before
        delta_A_err = (x_A_after - x_A_before).abs().mean().item()

        layer_data.append({
            "rel_delta": rel_delta,
            "cos_sim": cos,
            "delta_B_mean": delta_B.mean().item(),
            "delta_B_std": delta_B.std().item(),
            "delta_B_abs_mean": delta_B.abs().mean().item(),
            "delta_A_err": delta_A_err,
            "log_det": ld.mean().item(),
        })

    z_out = z_flat.reshape(B, C, H, W)
    if scale_block.apply_squeeze:
        z_out = scale_block._squeeze(z_out)
    return z_out, layer_data


def _collect_all_coupling_data(model, val_loader, device, n_batches):
    """Run manual forward through all scales, return per-layer stats across batches."""
    all_layer_data = {}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            h = model.conditioner(y)
            z = x.clone()

            for s_name, scale_block in [("scale1", model.scale1), ("scale2", model.scale2), ("scale3", model.scale3)]:
                z, layer_data = _manual_scale_forward(scale_block, z, h)
                for c_idx, data in enumerate(layer_data):
                    name = f"{s_name}_coupling{c_idx+1}"
                    if name not in all_layer_data:
                        all_layer_data[name] = []
                    all_layer_data[name].append(data)
                if s_name != "scale3":
                    z, _ = model._factor_out(z, factor_ratio=0.5)

    logger.info(f"  Captured data for {len(all_layer_data)} coupling layers")
    return all_layer_data


# ═══════════════════════════════════════════════════════════════════════
# CHECK 1 — log_det values
# ═══════════════════════════════════════════════════════════════════════
def check1_logdet_stats(model, val_loader, device, n_batches=5):
    logger.info("=" * 60)
    logger.info("[CHECK 1] log_det statistics")

    scale_logdets = {"scale1": [], "scale2": [], "scale3": [], "total": []}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            B = x.shape[0]
            log_det_total = torch.zeros(B, device=device)
            h = model.conditioner(y)
            model._cached_h = h
            z = x.clone()

            z, ld1 = model.scale1.forward(z, h, reverse=False)
            log_det_total += ld1
            scale_logdets["scale1"].append(ld1.cpu())
            z, _ = model._factor_out(z, factor_ratio=0.5)

            z, ld2 = model.scale2.forward(z, h, reverse=False)
            log_det_total += ld2
            scale_logdets["scale2"].append(ld2.cpu())
            z, _ = model._factor_out(z, factor_ratio=0.5)

            z, ld3 = model.scale3.forward(z, h, reverse=False)
            log_det_total += ld3
            scale_logdets["scale3"].append(ld3.cpu())
            scale_logdets["total"].append(log_det_total.cpu())

    results = {}
    for key in scale_logdets:
        all_vals = torch.cat(scale_logdets[key])
        stats = {"mean": all_vals.mean().item(), "std": all_vals.std().item(),
                 "min": all_vals.min().item(), "max": all_vals.max().item(),
                 "n_samples": len(all_vals)}
        results[key] = stats

        flag = ""
        if stats["std"] < 0.01:
            flag = " ⚠️ std < 0.01 → CONSTANT"
        if abs(stats["mean"]) > 1e4:
            flag += " ⚠️ |mean| > 1e4 → EXPLODING"
        logger.info(f"  {key:8s}: mean={stats['mean']:10.2f}, std={stats['std']:8.2f}, "
                     f"min={stats['min']:10.2f}, max={stats['max']:10.2f}{flag}")

    csv_path = os.path.join(OUTPUT_DIR, "check1_logdet_stats.csv")
    try:
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["scale", "mean", "std", "min", "max", "n_samples"])
            for key, s in results.items():
                w.writerow([key, s["mean"], s["std"], s["min"], s["max"], s["n_samples"]])
        logger.info(f"  Saved: {csv_path}")
    except IOError as e:
        logger.error(f"Failed to write {csv_path}: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK 2 — Sample quality
# ═══════════════════════════════════════════════════════════════════════
def check2_sample_quality(model, val_loader, device, n_samples=64):
    logger.info("=" * 60)
    logger.info("[CHECK 2] Sample quality")

    batch = next(iter(val_loader))
    y = batch[1][:n_samples].to(device)
    x_clean = batch[0][:n_samples].to(device)

    with torch.no_grad():
        try:
            samples = model.sample(n_samples=y.shape[0], y=y)
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise

    results = {"mean": samples.mean().item(), "std": samples.std().item(),
               "min": samples.min().item(), "max": samples.max().item(),
               "n_nan": torch.isnan(samples).sum().item(),
               "n_inf": torch.isinf(samples).sum().item(), "n_samples": n_samples}

    if results["n_nan"] > 0:
        logger.error(f"  ❌ FATAL: {results['n_nan']} NaN values in samples")
    if results["std"] < 1e-6:
        logger.error(f"  ❌ FATAL: std={results['std']:.2e} → collapsed samples")
    else:
        logger.info(f"  mean={results['mean']:.4f}, std={results['std']:.4f}, "
                     f"range=[{results['min']:.4f}, {results['max']:.4f}]")

    n_show = min(8, n_samples)
    fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))
    fig.suptitle(f"Check 2 — Sample Quality ({VERSION})", fontsize=12)
    for j in range(n_show):
        axes[0, j].imshow(x_clean[j, 0].cpu().numpy(), cmap='gray'); axes[0, j].axis('off')
        axes[0, j].set_title("x" if j == 0 else "", fontsize=8)
        y_img = y[j, 0] if y[j].ndim == 3 else y[j]
        axes[1, j].imshow(y_img.cpu().numpy(), cmap='gray'); axes[1, j].axis('off')
        axes[1, j].set_title("y" if j == 0 else "", fontsize=8)
        axes[2, j].imshow(samples[j, 0].cpu().numpy(), cmap='gray'); axes[2, j].axis('off')
        axes[2, j].set_title("x̂" if j == 0 else "", fontsize=8)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(OUTPUT_DIR, "check2_sample_grid.png"), dpi=150, bbox_inches='tight')
        logger.info(f"  Saved: check2_sample_grid.png")
    except Exception as e:
        logger.error(f"Failed to save check2_sample_grid.png: {e}")
    plt.close()

    try:
        with open(os.path.join(OUTPUT_DIR, "check2_sample_stats.csv"), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(results.keys()); w.writerow(results.values())
        logger.info(f"  Saved: check2_sample_stats.csv")
    except IOError as e:
        logger.error(f"Failed to write check2_sample_stats.csv: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK 3 — Coupling layer transformation delta (direct forward)
# ═══════════════════════════════════════════════════════════════════════
def check3_coupling_delta(model, val_loader, device, n_batches=3):
    """Uses direct manual forward (not hooks) — ScaleBlock calls coupling.forward()
    directly, bypassing __call__ and any registered hooks."""
    logger.info("=" * 60)
    logger.info("[CHECK 3] Coupling transformation delta (direct forward)")

    all_layer_data = _collect_all_coupling_data(model, val_loader, device, n_batches)
    if not all_layer_data:
        logger.error("  ❌ No coupling data captured")
        return {}

    results = {}
    for name in sorted(all_layer_data.keys()):
        batches = all_layer_data[name]
        stats = {
            "rel_delta_mean": np.mean([b["rel_delta"] for b in batches]),
            "rel_delta_std": np.std([b["rel_delta"] for b in batches]),
            "cos_sim_mean": np.mean([b["cos_sim"] for b in batches]),
            "cos_sim_std": np.std([b["cos_sim"] for b in batches]),
        }
        results[name] = stats
        flag = ""
        if stats["rel_delta_mean"] < 1e-4 and stats["cos_sim_mean"] > 0.9999:
            flag = " ⚠️ IDENTITY"
        elif stats["rel_delta_mean"] < 0.01:
            flag = " ⚠️ NEAR-IDENTITY"
        logger.info(f"  {name:22s}: rel_delta={stats['rel_delta_mean']:.6f}±{stats['rel_delta_std']:.6f}, "
                     f"cos_sim={stats['cos_sim_mean']:.6f}±{stats['cos_sim_std']:.6f}{flag}")

    try:
        with open(os.path.join(OUTPUT_DIR, "check3_coupling_delta.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["layer", "rel_delta_mean", "rel_delta_std", "cos_sim_mean", "cos_sim_std"])
            for name, s in results.items():
                w.writerow([name, s["rel_delta_mean"], s["rel_delta_std"], s["cos_sim_mean"], s["cos_sim_std"]])
        logger.info(f"  Saved: check3_coupling_delta.csv")
    except IOError as e:
        logger.error(f"Failed to write check3_coupling_delta.csv: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK A — Per-scale log_det violin plot
# ═══════════════════════════════════════════════════════════════════════
def checkA_logdet_violin(model, val_loader, device, n_batches=5):
    logger.info("=" * 60)
    logger.info("[CHECK A] Per-scale log_det violin plot")

    scale_logdets = {"Scale 1": [], "Scale 2": [], "Scale 3": []}
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            h = model.conditioner(y)
            z = x.clone()
            z, ld1 = model.scale1.forward(z, h, reverse=False)
            scale_logdets["Scale 1"].append(ld1.cpu())
            z, _ = model._factor_out(z, factor_ratio=0.5)
            z, ld2 = model.scale2.forward(z, h, reverse=False)
            scale_logdets["Scale 2"].append(ld2.cpu())
            z, _ = model._factor_out(z, factor_ratio=0.5)
            z, ld3 = model.scale3.forward(z, h, reverse=False)
            scale_logdets["Scale 3"].append(ld3.cpu())

    data_for_violin = []
    for key in ["Scale 1", "Scale 2", "Scale 3"]:
        vals = torch.cat(scale_logdets[key]).numpy()
        data_for_violin.append(vals)
        logger.info(f"  {key}: std={np.std(vals):.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    vp = ax.violinplot(data_for_violin, positions=[1, 2, 3], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["Scale 1", "Scale 2", "Scale 3"])
    ax.set_ylabel("log_det"); ax.set_title(f"Check A — Per-Scale log_det ({VERSION})")
    ax.grid(True, alpha=0.3)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(['#4C72B0', '#55A868', '#C44E52'][i]); body.set_alpha(0.7)
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(OUTPUT_DIR, "checkA_logdet_violin.png"), dpi=150, bbox_inches='tight')
        logger.info(f"  Saved: checkA_logdet_violin.png")
    except Exception as e:
        logger.error(f"Failed to save checkA_logdet_violin.png: {e}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# CHECK B — Latent z histogram vs N(0,1) — robust to extreme values
# ═══════════════════════════════════════════════════════════════════════
def checkB_latent_z_histogram(model, val_loader, device, n_batches=5):
    logger.info("=" * 60)
    logger.info("[CHECK B] Latent z histogram vs N(0,1)")

    z_final_all, z_fact1_all, z_fact2_all = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            x, y = batch[0].to(device), batch[1].to(device)
            z_final, z_factored_list, _, _ = model.forward(x, y, compute_h=True)
            z_final_all.append(z_final.cpu().reshape(-1))
            z_fact1_all.append(z_factored_list[0].cpu().reshape(-1))
            z_fact2_all.append(z_factored_list[1].cpu().reshape(-1))

    z_collections = {
        "z_final": torch.cat(z_final_all).numpy(),
        "z_factor1": torch.cat(z_fact1_all).numpy(),
        "z_factor2": torch.cat(z_fact2_all).numpy(),
    }

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Check B — Latent z vs N(0,1) ({VERSION})", fontsize=12)

    for ax_idx, (name, z_vals) in enumerate(z_collections.items()):
        z_mean = float(np.mean(z_vals))
        z_std = float(np.std(z_vals))
        results[name] = {"mean": z_mean, "std": z_std}

        flag = ""
        if abs(z_mean) > 1.0:
            flag += " ⚠️ mean shifted"
        if abs(z_std - 1.0) > 1.0:
            flag += " ⚠️ std misaligned"
        logger.info(f"  {name}: mean={z_mean:.4f}, std={z_std:.4f}{flag}")

        ax = axes[ax_idx]

        # Use percentile-based clipping for extreme z values
        p1, p99 = float(np.percentile(z_vals, 1)), float(np.percentile(z_vals, 99))
        margin = max(abs(p99 - p1) * 0.1, 1.0)
        clip_lo = p1 - margin
        clip_hi = p99 + margin
        z_clipped = np.clip(z_vals, clip_lo, clip_hi)

        # Guard against degenerate histogram (all same value after clip)
        if np.std(z_clipped) < 1e-10:
            ax.text(0.5, 0.5, f"Degenerate z\nμ={z_mean:.1f}\nσ={z_std:.1f}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{name}\nμ={z_mean:.1f}, σ={z_std:.1f}")
            continue

        try:
            ax.hist(z_clipped, bins=100, density=True, alpha=0.6, color='steelblue', label=name)
        except Exception as e:
            logger.error(f"  Histogram failed for {name}: {e}")
            ax.text(0.5, 0.5, f"Plot failed:\n{e}", ha='center', va='center', transform=ax.transAxes)
            continue

        # N(0,1) reference — only visible if z is near standard normal
        x_ref = np.linspace(clip_lo, clip_hi, 300)
        y_ref = np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi)
        ax.plot(x_ref, y_ref, 'r--', lw=2, label='N(0,1)')

        ax.set_title(f"{name}\nμ={z_mean:.1f}, σ={z_std:.1f}")
        ax.set_xlabel("z value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "checkB_latent_z_hist.png")
    try:
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved: {png_path}")
    except Exception as e:
        logger.error(f"Failed to save {png_path}: {e}")
    plt.close()
    return results


# ═══════════════════════════════════════════════════════════════════════
# CHECK C — s/t network output stats (direct forward, no hooks)
# ═══════════════════════════════════════════════════════════════════════
def checkC_st_network_stats(model, val_loader, device, n_batches=3):
    """Uses direct manual forward (not hooks) — same reason as check3."""
    logger.info("=" * 60)
    logger.info("[CHECK C] s/t network output stats (direct forward)")

    all_layer_data = _collect_all_coupling_data(model, val_loader, device, n_batches)
    if not all_layer_data:
        logger.error("  ❌ No coupling data captured")
        return {}

    results = {}
    for name in sorted(all_layer_data.keys()):
        batches = all_layer_data[name]
        stats = {
            "delta_B_mean": np.mean([b["delta_B_mean"] for b in batches]),
            "delta_B_std": np.mean([b["delta_B_std"] for b in batches]),
            "delta_B_abs_mean": np.mean([b["delta_B_abs_mean"] for b in batches]),
            "delta_A_err": np.mean([b["delta_A_err"] for b in batches]),
        }
        results[name] = stats
        flag = ""
        if stats["delta_B_abs_mean"] < 1e-4:
            flag = " ⚠️ DEAD COUPLING"
        if stats["delta_A_err"] > 1e-5:
            flag += " ⚠️ PASSTHROUGH CORRUPTED"
        logger.info(f"  {name:22s}: |ΔB|={stats['delta_B_abs_mean']:.6f}, "
                     f"ΔB_std={stats['delta_B_std']:.6f}, ΔA_err={stats['delta_A_err']:.2e}{flag}")

    try:
        with open(os.path.join(OUTPUT_DIR, "checkC_st_network_stats.csv"), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["layer", "delta_B_mean", "delta_B_std", "delta_B_abs_mean", "delta_A_err"])
            for name, s in results.items():
                w.writerow([name, s["delta_B_mean"], s["delta_B_std"], s["delta_B_abs_mean"], s["delta_A_err"]])
        logger.info(f"  Saved: checkC_st_network_stats.csv")
    except IOError as e:
        logger.error(f"Failed to write checkC_st_network_stats.csv: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
def write_summary(all_results):
    logger.info("=" * 60)
    logger.info("[SUMMARY]")

    summary = {"version": VERSION, "timestamp": datetime.now().isoformat(), "checks": {}}

    # Check 1
    if "check1" in all_results:
        c1 = all_results["check1"]
        total = c1.get("total", {})
        status, issues = "PASS", []
        if total.get("std", 1.0) < 0.01:
            status = "FAIL"; issues.append("log_det std < 0.01 (constant)")
        if abs(total.get("mean", 0)) > 1e4:
            status = "WARN"; issues.append(f"|mean|={abs(total.get('mean', 0)):.0f} (exploding)")
        summary["checks"]["check1_logdet"] = {"status": status, "issues": issues, "data": c1}

    # Check 2
    if "check2" in all_results:
        c2 = all_results["check2"]
        status, issues = "PASS", []
        if c2.get("n_nan", 0) > 0:
            status = "FAIL"; issues.append(f"{c2['n_nan']} NaN in samples")
        if c2.get("std", 1.0) < 1e-6:
            status = "FAIL"; issues.append("std < 1e-6 (collapsed)")
        summary["checks"]["check2_samples"] = {"status": status, "issues": issues, "data": c2}

    # Check 3 — guard empty
    if "check3" in all_results:
        c3 = all_results["check3"]
        if not c3:
            summary["checks"]["check3_coupling_delta"] = {
                "status": "FAIL", "issues": ["No data captured"], "n_identity": -1, "n_near_identity": -1}
        else:
            n_id = sum(1 for v in c3.values() if v["rel_delta_mean"] < 1e-4)
            n_near = sum(1 for v in c3.values() if v["rel_delta_mean"] < 0.01)
            status = "FAIL" if n_id > 0 else ("WARN" if n_near > 3 else "PASS")
            issues = []
            if n_id > 0: issues.append(f"{n_id}/9 identity")
            if n_near > 3: issues.append(f"{n_near}/9 near-identity")
            summary["checks"]["check3_coupling_delta"] = {
                "status": status, "issues": issues, "n_identity": n_id, "n_near_identity": n_near}

    # Check B
    if "checkB" in all_results:
        cB = all_results["checkB"]
        status, issues = "PASS", []
        for name, s in cB.items():
            if abs(s["mean"]) > 1.0:
                status = "WARN"; issues.append(f"{name} mean={s['mean']:.3f}")
            if abs(s["std"] - 1.0) > 1.0:
                status = "WARN"; issues.append(f"{name} std={s['std']:.3f}")
        summary["checks"]["checkB_latent_z"] = {"status": status, "issues": issues, "data": cB}

    # Check C — guard empty
    if "checkC" in all_results:
        cC = all_results["checkC"]
        if not cC:
            summary["checks"]["checkC_st_stats"] = {
                "status": "FAIL", "issues": ["No data captured"], "n_dead": -1}
        else:
            n_dead = sum(1 for v in cC.values() if v["delta_B_abs_mean"] < 1e-4)
            status = "FAIL" if n_dead > 0 else "PASS"
            issues = [f"{n_dead}/9 dead"] if n_dead > 0 else []
            summary["checks"]["checkC_st_stats"] = {"status": status, "issues": issues, "n_dead": n_dead}

    statuses = [v["status"] for v in summary["checks"].values()]
    summary["overall"] = "FAIL" if "FAIL" in statuses else ("WARN" if "WARN" in statuses else "PASS")

    logger.info(f"  Overall: {summary['overall']}")
    for cn, cd in summary["checks"].items():
        logger.info(f"    {cn}: {cd['status']} — {cd.get('issues', [])}")

    try:
        with open(os.path.join(OUTPUT_DIR, "diagnostic_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"  Saved: diagnostic_summary.json")
    except IOError as e:
        logger.error(f"Failed to write diagnostic_summary.json: {e}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=f"{VERSION} — RealNVP flat-line diagnostic")
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./data/preprocessed')
    parser.add_argument('--n_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    ckpt_path = None if args.no_checkpoint else args.checkpoint
    if ckpt_path is None and not args.no_checkpoint:
        logger.error("Must provide --checkpoint or --no-checkpoint")
        sys.exit(1)

    ensure_output_dir()
    model, val_loader = load_model_and_data(ckpt_path, args.data_dir, device,
                                             n_val_batches=args.n_batches, batch_size=args.batch_size)

    all_results = {}
    all_results["check1"] = check1_logdet_stats(model, val_loader, device, n_batches=args.n_batches)
    all_results["check2"] = check2_sample_quality(model, val_loader, device, n_samples=64)
    all_results["check3"] = check3_coupling_delta(model, val_loader, device, n_batches=min(3, args.n_batches))
    checkA_logdet_violin(model, val_loader, device, n_batches=args.n_batches)
    all_results["checkB"] = checkB_latent_z_histogram(model, val_loader, device, n_batches=args.n_batches)
    all_results["checkC"] = checkC_st_network_stats(model, val_loader, device, n_batches=min(3, args.n_batches))
    write_summary(all_results)

    logger.info("=" * 60)
    logger.info(f"{VERSION} — Complete. Results in {OUTPUT_DIR}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
