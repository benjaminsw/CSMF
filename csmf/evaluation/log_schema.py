# =============================================================================
# Version: DIAG-REORG-LogSchema-v1.7 | Abbr: LS
# Description: TypedDict contracts and validators for epoch_logs dicts produced
#              by CSMF-MAIN train_stage_A/B/C(). Single source of truth for
#              all diagnostic scripts (SA-DIAG, SB-DIAG, SC-DIAG) and
#              metric_utils (MU). Import and call validate_*() at the top of
#              every diagnostic run() to catch missing keys early.
# Changelog:
#   v1.7 (2026-04-19): [NEFF-REG] Add optional Stage B keys: neff_reg_loss
#                      (List[float], λ_neff·max(0,Neff_target−Neff) per epoch),
#                      lambda_cons_eff / lambda_trans_eff / lambda_cal_eff
#                      (List[float], effective annealed lambda per epoch) — all
#                      populated by HYBRID v1.10.0 forward() loss_dict and
#                      accumulated in CSMF-MAIN train_stage_B(); consumed by
#                      SB-DIAG v1.5 P_neff_reg and P_anneal_lambdas.
#                      [PRE-PROX][SC-TRANS-CAL] Add optional Stage C keys:
#                      cons_pre_loss (List[float], λ_cons_pre·‖A(x̂_mix)−y‖² per epoch),
#                      residual_pre_prox / residual_post_prox (List[float], raw
#                      ‖A(x̂)−y‖² before/after prox per epoch),
#                      trans_loss_c / cal_loss_c (List[float], SW2 and ES terms
#                      per epoch) — all from HYBRID v1.10.0 forward_stage_c()
#                      loss_dict; consumed by SC-DIAG v2.5 P_pre_post_prox,
#                      P_trans_cal_c, and P_loss_components_c1.
#   v1.6 (2026-04-17): [SC-EPOCH-LOGS] Add optional cons_c1, img_loss,
#                      alive_penalty, neff_c1 to StageCEpochLogs — populated
#                      from HybridLoss.forward_stage_c() loss_dict per epoch
#                      by CSMF-MAIN v1.3.15 train_stage_C(); consumed by
#                      SC-DIAG v2.4 P_loss_components_c1 and P_neff_c1.
#                      [PROX-C-ACTIVATE] Add optional prox_applied_rate —
#                      List[float] fraction of batches where prox_fn fired per
#                      epoch; consumed by SC-DIAG v2.4 P_prox_rate; extends
#                      _STAGE_C_OPTIONAL accordingly.
#   v1.5 (2026-04-17): [PROX-T] Add optional prox_residuals and prox_nll keys
#                      to StageCEpochLogs — prox_residuals: Dict[str,List[float]]
#                      maps str(T) to per-step residuals [||Ax^(t)-y||² t=0..T];
#                      prox_nll: Dict[str,float] maps str(T) to mean NLL baseline;
#                      prox_sample_std: Dict[str,float] maps "pre"/"post" to sample
#                      std; extend _STAGE_C_OPTIONAL; consumed by SA-DIAG v1.6
#                      P_PROX1, P_PROX2, P_PROX3 via collect_prox_diagnostics (MU v1.3)
#   v1.4 (2026-04-11): [LOGDET-DIAG] Add optional log_det_mean and log_det_std
#                      keys to StageAExpertLogs — List[float] of per-epoch
#                      mean/std log|det J| from single val batch, produced by
#                      CSMF-MAIN v1.7 train_stage_A(); extend _STAGE_A_OPTIONAL
#                      accordingly; consumed by SA-DIAG v1.5 P15 temporal plot
#   v1.3 (2026-04-09): [PATCH-SA-SCW] Add optional gap_penalty key to
#                      StageAExpertLogs — List[float] of mean gap penalty
#                      per epoch, produced by CSMF-MAIN v1.5 train_stage_A();
#                      extend _STAGE_A_OPTIONAL accordingly; consumed by
#                      SA-DIAG _build_summary() for JSON serialisation
#   v1.2 (2026-04-09): [PATCH-SA-SCW] Add optional soft_weights key to
#                      StageAExpertLogs — List[float] of mean competition weight
#                      w_k per epoch, produced by CSMF-MAIN v1.4 train_stage_A();
#                      extend _STAGE_A_OPTIONAL accordingly; consumed by
#                      SA-DIAG v1.3 P10 (weight-over-epochs) and P11 (gap hist)
#   v1.1 (2026-04-07): [DIAG-OUTPUT] Add optional per-component loss keys
#                      (nll_loss, cons_loss, trans_loss, cal_loss) to
#                      StageBEpochLogs and StageCEpochLogs; extend
#                      _STAGE_B/C_OPTIONAL sentinel sets accordingly;
#                      consumed by SB-DIAG v1.1 and SC-DIAG v2.1 P_loss_components
#   v1.0 (2026-04-04): Initial implementation — StageAExpertLogs,
#                      StageBEpochLogs, StageCEpochLogs TypedDicts;
#                      validate_stage_a/b/c_logs() returning (ok, missing_keys);
#                      REQUIRED_* sentinel sets for each stage; all validators
#                      log warnings for optional keys and errors for required keys
# Dependencies: Python >= 3.8 (TypedDict), logging
# =============================================================================

import logging
from typing import Any, Dict, List, Optional, Tuple

# Python 3.8+: TypedDict available in typing
from typing import TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# Stage A — per-expert sub-dict inside epoch_logs[<expert_name>]
# Produced by: CSMF-MAIN train_stage_A() (v1.3.19+)
# =============================================================================

class StageAExpertLogs(TypedDict, total=False):
    """
    Per-expert epoch log dict stored under epoch_logs[<expert_name>].

    Required keys:
        train_nll : List[float]  — mean train NLL per epoch
        val_nll   : List[float]  — mean val NLL per epoch

    Optional keys (may be absent in older checkpoints):
        inv_err      : List[float]  — mean ‖x - f⁻¹(f(x))‖ per epoch
        fi_a         : List[float]  — Fisher Information Option A scalar per epoch
                                      (requires CSMF-MAIN v1.3.19+)
        soft_weights : List[float]  — mean competition weight w_k per epoch
                                      (requires CSMF-MAIN v1.4+, PATCH-SA-SCW)
        gap_penalty  : List[float]  — mean gap penalty term per epoch
                                      (requires CSMF-MAIN v1.5+, PATCH-SA-SCW)
        log_det_mean : List[float]  — mean log|det J| per epoch (single val batch)
                                      (requires CSMF-MAIN v1.7+, LOGDET-DIAG)
        log_det_std  : List[float]  — std  log|det J| per epoch (single val batch)
                                      (requires CSMF-MAIN v1.7+, LOGDET-DIAG)
    """
    train_nll    : List[float]
    val_nll      : List[float]
    inv_err      : List[float]   # optional
    fi_a         : List[float]   # optional
    soft_weights : List[float]   # optional — PATCH-SA-SCW, CSMF-MAIN v1.4+
    gap_penalty  : List[float]   # optional — PATCH-SA-SCW, CSMF-MAIN v1.5+
    log_det_mean : List[float]   # optional — LOGDET-DIAG, CSMF-MAIN v1.7+
    log_det_std  : List[float]   # optional — LOGDET-DIAG, CSMF-MAIN v1.7+


# Required keys that validators enforce
_STAGE_A_REQUIRED: frozenset = frozenset({"train_nll", "val_nll"})
_STAGE_A_OPTIONAL: frozenset = frozenset({
    "inv_err", "fi_a", "soft_weights", "gap_penalty",
    "log_det_mean", "log_det_std",                       # [LOGDET-DIAG] v1.4
})


# =============================================================================
# Stage B — flat epoch_logs dict returned by train_stage_B()
# Produced by: CSMF-MAIN train_stage_B() (v1.3.23+ for tau key)
# =============================================================================

class StageBEpochLogs(TypedDict, total=False):
    """
    Flat epoch log dict returned by train_stage_B().

    Required keys:
        train_loss   : List[float]           — mean train hybrid loss per epoch
        neff         : List[float]           — effective number of experts per epoch
        gate_weights : List[List[float]]     — mean gate weight per expert per epoch
                                               shape: [epochs, K]

    Optional keys:
        val_loss     : List[float]           — mean val hybrid loss per epoch
                                               (absent if no val_loader provided)
        tau          : List[float]           — temperature per epoch
                                               (requires CSMF-MAIN v1.3.23+)
    """
    train_loss   : List[float]
    neff         : List[float]
    gate_weights : List[List[float]]
    val_loss     : List[float]        # optional
    tau          : List[float]        # optional — requires CSMF-MAIN v1.3.23+
    nll_loss     : List[float]        # optional — requires CSMF-MAIN v1.3.29+
    cons_loss    : List[float]        # optional — requires CSMF-MAIN v1.3.29+
    trans_loss   : List[float]        # optional — requires CSMF-MAIN v1.3.29+
    cal_loss     : List[float]        # optional — requires CSMF-MAIN v1.3.29+
    # [NEFF-REG] v1.7 — from HYBRID v1.10.0 forward() via CSMF-MAIN train_stage_B()
    neff_reg_loss     : List[float]   # optional — λ_neff·max(0,Neff_target−Neff) per epoch
    lambda_cons_eff   : List[float]   # optional — effective annealed λ_cons per epoch
    lambda_trans_eff  : List[float]   # optional — effective annealed λ_trans per epoch
    lambda_cal_eff    : List[float]   # optional — effective annealed λ_cal per epoch


_STAGE_B_REQUIRED: frozenset = frozenset({"train_loss", "neff", "gate_weights"})
_STAGE_B_OPTIONAL: frozenset = frozenset({
    "val_loss", "tau", "nll_loss", "cons_loss", "trans_loss", "cal_loss",
    "neff_reg_loss",                             # [NEFF-REG] v1.7
    "lambda_cons_eff", "lambda_trans_eff", "lambda_cal_eff",  # [NEFF-REG] v1.7
})


# =============================================================================
# Stage C — flat epoch_logs dict returned by train_stage_C()
# Produced by: CSMF-MAIN train_stage_C() (v1.3.22+)
# =============================================================================

class StageCEpochLogs(TypedDict, total=False):
    """
    Flat epoch log dict returned by train_stage_C().

    Required keys:
        train_loss     : List[float]          — mean train joint loss per epoch
        val_loss       : List[float]          — mean val joint loss per epoch
        neff           : List[float]          — effective number of experts per epoch

    Optional keys:
        tau            : List[float]          — temperature per epoch
        gate_weights   : List[List[float]]    — mean gate weight per expert per epoch
        residual       : List[float]          — mean ‖Ax̂ - y‖² per epoch
        recon_snapshots: List[Any]            — list of (y, x_hat) tensor pairs,
                                               one entry every recon_every epochs
                                               (requires CSMF-MAIN v1.3.22+)
    """
    train_loss      : List[float]
    val_loss        : List[float]
    neff            : List[float]
    tau             : List[float]         # optional
    gate_weights    : List[List[float]]   # optional
    residual        : List[float]         # optional
    recon_snapshots : List[Any]           # optional
    nll_loss        : List[float]         # optional — requires CSMF-MAIN v1.3.29+
    cons_loss       : List[float]         # optional — requires CSMF-MAIN v1.3.29+
    trans_loss      : List[float]         # optional — requires CSMF-MAIN v1.3.29+
    cal_loss        : List[float]         # optional — requires CSMF-MAIN v1.3.29+
    # [SC-EPOCH-LOGS] v1.6 — from HybridLoss.forward_stage_c() via CSMF-MAIN v1.3.15
    cons_c1         : List[float]         # optional — ‖A(x̂_mix)−y‖² per epoch
    img_loss        : List[float]         # optional — ‖x̂_mix−x‖² per epoch
    alive_penalty   : List[float]         # optional — max(0, 1.5−Neff) per epoch
    neff_c1         : List[float]         # optional — Neff from forward_stage_c per epoch
    # [PROX-C-ACTIVATE] v1.6 — prox application rate per epoch
    prox_applied_rate : List[float]       # optional — fraction of batches where prox fired
    # [PROX-T] v1.5 — proximal correction diagnostics (eval-time, not per-epoch)
    prox_residuals  : Dict[str, List[float]]   # optional — str(T) -> per-step residuals
    prox_nll        : Dict[str, float]          # optional — str(T) -> mean NLL baseline
    prox_sample_std : Dict[str, float]          # optional — "pre"/"post" -> sample std
    # [PRE-PROX][SC-TRANS-CAL] v1.7 — from HYBRID v1.10.0 forward_stage_c()
    cons_pre_loss      : List[float]      # optional — λ_cons_pre·‖A(x̂_mix)−y‖² per epoch
    residual_pre_prox  : List[float]      # optional — raw ‖A(x̂_mix)−y‖² before prox
    residual_post_prox : List[float]      # optional — raw ‖A(x̂_corr)−y‖² after prox
    trans_loss_c       : List[float]      # optional — SW2 term per epoch
    cal_loss_c         : List[float]      # optional — ES term per epoch


_STAGE_C_REQUIRED: frozenset = frozenset({"train_loss", "val_loss", "neff"})
_STAGE_C_OPTIONAL: frozenset = frozenset({
    "tau", "gate_weights", "residual", "recon_snapshots",
    "nll_loss", "cons_loss", "trans_loss", "cal_loss",
    "prox_residuals", "prox_nll", "prox_sample_std",     # [PROX-T] v1.5
    "cons_c1", "img_loss", "alive_penalty", "neff_c1",   # [SC-EPOCH-LOGS] v1.6
    "prox_applied_rate",                                  # [PROX-C-ACTIVATE] v1.6
    "cons_pre_loss", "residual_pre_prox", "residual_post_prox",  # [PRE-PROX] v1.7
    "trans_loss_c", "cal_loss_c",                         # [SC-TRANS-CAL] v1.7
})


# =============================================================================
# Validators
# =============================================================================

def validate_stage_a_logs(
    epoch_logs: Dict[str, Any],
    expert_names: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate Stage A epoch_logs structure.

    Args:
        epoch_logs   : dict mapping expert_name → per-expert log dict
        expert_names : if provided, also checks these keys exist in epoch_logs

    Returns:
        (ok, missing)
            ok      : True if all required keys present for all experts
            missing : list of "<expert>.<key>" strings that are missing
    """
    missing: List[str] = []

    if not isinstance(epoch_logs, dict):
        logger.error(
            "LS | validate_stage_a_logs: epoch_logs must be dict, "
            f"got {type(epoch_logs).__name__}"
        )
        return False, ["epoch_logs_not_dict"]

    # If expert_names provided, check top-level keys
    names_to_check = expert_names if expert_names else list(epoch_logs.keys())

    if not names_to_check:
        logger.error("LS | validate_stage_a_logs: epoch_logs is empty")
        return False, ["epoch_logs_empty"]

    for name in names_to_check:
        if name not in epoch_logs:
            logger.error(f"LS | validate_stage_a_logs: expert '{name}' missing from epoch_logs")
            missing.append(f"{name}.<expert_missing>")
            continue

        expert_log = epoch_logs[name]

        if not isinstance(expert_log, dict):
            logger.error(
                f"LS | validate_stage_a_logs: epoch_logs['{name}'] must be dict, "
                f"got {type(expert_log).__name__}"
            )
            missing.append(f"{name}.<not_dict>")
            continue

        for key in _STAGE_A_REQUIRED:
            if key not in expert_log:
                logger.error(
                    f"LS | validate_stage_a_logs: required key '{key}' "
                    f"missing for expert '{name}'"
                )
                missing.append(f"{name}.{key}")
            elif not isinstance(expert_log[key], list):
                logger.error(
                    f"LS | validate_stage_a_logs: epoch_logs['{name}']['{key}'] "
                    f"must be list, got {type(expert_log[key]).__name__}"
                )
                missing.append(f"{name}.{key}<not_list>")

        for key in _STAGE_A_OPTIONAL:
            if key not in expert_log:
                logger.warning(
                    f"LS | validate_stage_a_logs: optional key '{key}' "
                    f"absent for expert '{name}' — corresponding plot will be skipped"
                )

    ok = len(missing) == 0
    if ok:
        logger.info(
            f"LS | validate_stage_a_logs: OK — {len(names_to_check)} expert(s) validated"
        )
    else:
        logger.error(
            f"LS | validate_stage_a_logs: FAILED — {len(missing)} missing item(s): {missing}"
        )
    return ok, missing


def validate_stage_b_logs(
    epoch_logs: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate Stage B epoch_logs structure (flat dict).

    Args:
        epoch_logs : flat dict from train_stage_B()

    Returns:
        (ok, missing)
    """
    missing: List[str] = []

    if not isinstance(epoch_logs, dict):
        logger.error(
            "LS | validate_stage_b_logs: epoch_logs must be dict, "
            f"got {type(epoch_logs).__name__}"
        )
        return False, ["epoch_logs_not_dict"]

    for key in _STAGE_B_REQUIRED:
        if key not in epoch_logs:
            logger.error(
                f"LS | validate_stage_b_logs: required key '{key}' missing"
            )
            missing.append(key)
        elif not isinstance(epoch_logs[key], list):
            logger.error(
                f"LS | validate_stage_b_logs: epoch_logs['{key}'] must be list, "
                f"got {type(epoch_logs[key]).__name__}"
            )
            missing.append(f"{key}<not_list>")

    for key in _STAGE_B_OPTIONAL:
        if key not in epoch_logs:
            logger.warning(
                f"LS | validate_stage_b_logs: optional key '{key}' absent"
                + (" — tau requires CSMF-MAIN v1.3.23+" if key == "tau" else "")
                + " — corresponding plot will be skipped"
            )

    ok = len(missing) == 0
    if ok:
        logger.info("LS | validate_stage_b_logs: OK")
    else:
        logger.error(
            f"LS | validate_stage_b_logs: FAILED — {len(missing)} missing item(s): {missing}"
        )
    return ok, missing


def validate_stage_c_logs(
    epoch_logs: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate Stage C epoch_logs structure (flat dict).

    Args:
        epoch_logs : flat dict from train_stage_C()

    Returns:
        (ok, missing)
    """
    missing: List[str] = []

    if not isinstance(epoch_logs, dict):
        logger.error(
            "LS | validate_stage_c_logs: epoch_logs must be dict, "
            f"got {type(epoch_logs).__name__}"
        )
        return False, ["epoch_logs_not_dict"]

    for key in _STAGE_C_REQUIRED:
        if key not in epoch_logs:
            logger.error(
                f"LS | validate_stage_c_logs: required key '{key}' missing"
            )
            missing.append(key)
        elif not isinstance(epoch_logs[key], list):
            logger.error(
                f"LS | validate_stage_c_logs: epoch_logs['{key}'] must be list, "
                f"got {type(epoch_logs[key]).__name__}"
            )
            missing.append(f"{key}<not_list>")

    for key in _STAGE_C_OPTIONAL:
        if key not in epoch_logs:
            logger.warning(
                f"LS | validate_stage_c_logs: optional key '{key}' absent"
                + (" — requires CSMF-MAIN v1.3.22+" if key == "recon_snapshots" else "")
                + (" — requires PROX-T + MU v1.3 collect_prox_diagnostics()"
                   if key in {"prox_residuals", "prox_nll", "prox_sample_std"} else "")
                + " — corresponding plot will be skipped"
            )

    ok = len(missing) == 0
    if ok:
        logger.info("LS | validate_stage_c_logs: OK")
    else:
        logger.error(
            f"LS | validate_stage_c_logs: FAILED — {len(missing)} missing item(s): {missing}"
        )
    return ok, missing


# =============================================================================
# Convenience: validate and return missing optional keys (for conditional plots)
# =============================================================================

def available_optional_keys_a(epoch_logs: Dict[str, Any], expert_name: str) -> List[str]:
    """Return list of optional Stage A keys present for a given expert."""
    expert_log = epoch_logs.get(expert_name, {})
    return [k for k in _STAGE_A_OPTIONAL if k in expert_log]


def available_optional_keys_b(epoch_logs: Dict[str, Any]) -> List[str]:
    """Return list of optional Stage B keys present in epoch_logs."""
    return [k for k in _STAGE_B_OPTIONAL if k in epoch_logs]


def available_optional_keys_c(epoch_logs: Dict[str, Any]) -> List[str]:
    """Return list of optional Stage C keys present in epoch_logs."""
    return [k for k in _STAGE_C_OPTIONAL if k in epoch_logs]
