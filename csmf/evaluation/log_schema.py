# =============================================================================
# Version: DIAG-REORG-LogSchema-v1.1 | Abbr: LS
# Description: TypedDict contracts and validators for epoch_logs dicts produced
#              by CSMF-MAIN train_stage_A/B/C(). Single source of truth for
#              all diagnostic scripts (SA-DIAG, SB-DIAG, SC-DIAG) and
#              metric_utils (MU). Import and call validate_*() at the top of
#              every diagnostic run() to catch missing keys early.
# Changelog:
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
        inv_err   : List[float]  — mean ‖x - f⁻¹(f(x))‖ per epoch
        fi_a      : List[float]  — Fisher Information Option A scalar per epoch
                                   (requires CSMF-MAIN v1.3.19+)
    """
    train_nll : List[float]
    val_nll   : List[float]
    inv_err   : List[float]   # optional
    fi_a      : List[float]   # optional


# Required keys that validators enforce
_STAGE_A_REQUIRED: frozenset = frozenset({"train_nll", "val_nll"})
_STAGE_A_OPTIONAL: frozenset = frozenset({"inv_err", "fi_a"})


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


_STAGE_B_REQUIRED: frozenset = frozenset({"train_loss", "neff", "gate_weights"})
_STAGE_B_OPTIONAL: frozenset = frozenset({
    "val_loss", "tau", "nll_loss", "cons_loss", "trans_loss", "cal_loss"
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


_STAGE_C_REQUIRED: frozenset = frozenset({"train_loss", "val_loss", "neff"})
_STAGE_C_OPTIONAL: frozenset = frozenset({
    "tau", "gate_weights", "residual", "recon_snapshots",
    "nll_loss", "cons_loss", "trans_loss", "cal_loss",
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
