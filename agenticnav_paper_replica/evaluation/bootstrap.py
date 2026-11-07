"""Statistical evaluation and bootstrap confidence interval computation for AgenticNav."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    mean: float
    ci_lower: float
    ci_upper: float
    resample_count: int
    alpha: float = 0.05

    def contains(self, target_value: float) -> bool:
        return self.ci_lower <= target_value <= self.ci_upper


def compute_bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> BootstrapResult:
    """Compute non-parametric bootstrap confidence interval for the mean.

    Args:
        values: 1D array of per-episode metrics (e.g. SR indicators in {0, 1} or SPL in [0, 1]).
        n_resamples: Number of bootstrap resamples (default 10,000).
        alpha: Significance level (default 0.05 for 95% CI).
        random_seed: Seed for reproducibility.

    Returns:
        BootstrapResult with mean and CI bounds.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        raise ValueError("Cannot bootstrap empty array.")

    rng = np.random.default_rng(random_seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    sample_mean = float(np.mean(arr))

    return BootstrapResult(
        mean=sample_mean,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        resample_count=n_resamples,
        alpha=alpha,
    )


def verify_consistency_criterion(
    sr_per_episode: np.ndarray,
    spl_per_episode: np.ndarray,
    target_sr: float = 76.0,
    target_spl: float = 66.50,
    n_resamples: int = 10000,
) -> Dict[str, Any]:
    """Evaluates the primary Consistency Acceptance Rule (a).

    Acceptance Rule:
        76.0% in CI_95%(SR) AND 66.50% in CI_95%(SPL)
    """
    # Scale values to percentages if provided in [0, 1]
    sr_pct = sr_per_episode * 100.0 if np.max(sr_per_episode) <= 1.0 else sr_per_episode
    spl_pct = spl_per_episode * 100.0 if np.max(spl_per_episode) <= 1.0 else spl_per_episode

    ci_sr = compute_bootstrap_ci(sr_pct, n_resamples=n_resamples)
    ci_spl = compute_bootstrap_ci(spl_pct, n_resamples=n_resamples)

    sr_consistent = ci_sr.contains(target_sr)
    spl_consistent = ci_spl.contains(target_spl)
    overall_pass = sr_consistent and spl_consistent

    return {
        "overall_replication_pass": overall_pass,
        "sr": {
            "mean": round(ci_sr.mean, 2),
            "ci_95": [round(ci_sr.ci_lower, 2), round(ci_sr.ci_upper, 2)],
            "target": target_sr,
            "consistent": sr_consistent,
        },
        "spl": {
            "mean": round(ci_spl.mean, 2),
            "ci_95": [round(ci_spl.ci_lower, 2), round(ci_spl.ci_upper, 2)],
            "target": target_spl,
            "consistent": spl_consistent,
        },
    }


def paired_bootstrap_test(
    full_vals: np.ndarray,
    ablation_vals: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Non-parametric paired bootstrap hypothesis test for delta = (full - ablation) > 0.

    Applicable to both binary single-run ({0, 1}) and continuous multi-seed ({0, 1/3, 2/3, 1}) episode scores.

    Returns:
        Dict containing observed mean diff, 95% bootstrap CI, and one-sided p-value.
    """
    f = np.asarray(full_vals, dtype=np.float64)
    a = np.asarray(ablation_vals, dtype=np.float64)
    if len(f) != len(a):
        raise ValueError("Lengths of full and ablation arrays must match for paired test.")

    n = len(f)
    diffs = f - a
    obs_diff = float(np.mean(diffs))

    rng = np.random.default_rng(random_seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_diff_means = np.mean(diffs[boot_indices], axis=1)

    ci_lower = float(np.percentile(boot_diff_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_diff_means, 100.0 * (1.0 - alpha / 2.0)))

    # Empirical one-sided p-value for H0: delta <= 0 (proportion of resamples <= 0)
    # With continuity correction of +1 / (B + 1)
    p_val = float((np.sum(boot_diff_means <= 0.0) + 1.0) / (n_resamples + 1.0))

    return {
        "mean_diff": obs_diff,
        "ci_95": [ci_lower, ci_upper],
        "p_value": p_val,
        "significant": (p_val < alpha),
    }


def holm_bonferroni_correction(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Applies step-down Holm-Bonferroni correction across multiple hypothesis tests.

    Args:
        p_values: Dict mapping test_name -> raw p-value.
        alpha: Family-wise error rate (default 0.05).

    Returns:
        Dict mapping test_name -> {raw_p, adjusted_p, significant, rank}.
    """
    sorted_tests = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_tests)
    results = {}
    cum_max_p = 0.0

    for rank, (name, raw_p) in enumerate(sorted_tests, start=1):
        multiplier = m - rank + 1
        adj_p = min(1.0, raw_p * multiplier)
        adj_p = max(adj_p, cum_max_p)  # Enforce monotonicity
        cum_max_p = adj_p

        results[name] = {
            "raw_p": float(raw_p),
            "adjusted_p": float(round(adj_p, 5)),
            "significant": bool(adj_p < alpha),
            "rank": rank,
        }

    return results


def compare_paired_ablation(
    full_sr: np.ndarray,
    ablation_sr: np.ndarray,
    full_spl: np.ndarray,
    ablation_spl: np.ndarray,
    n_resamples: int = 10000,
) -> Dict[str, Any]:
    """Runs paired bootstrap test along with McNemar and Wilcoxon signed-rank tests."""
    f_sr = np.asarray(full_sr, dtype=np.float64)
    a_sr = np.asarray(ablation_sr, dtype=np.float64)
    f_spl = np.asarray(full_spl, dtype=np.float64)
    a_spl = np.asarray(ablation_spl, dtype=np.float64)

    # Paired bootstrap tests
    boot_sr = paired_bootstrap_test(f_sr, a_sr, n_resamples=n_resamples)
    boot_spl = paired_bootstrap_test(f_spl, a_spl, n_resamples=n_resamples)

    # McNemar test for binary paired SR
    b = int(np.sum((f_sr == 1) & (a_sr == 0)))
    c = int(np.sum((f_sr == 0) & (a_sr == 1)))

    if (b + c) > 0:
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_val_mcnemar = float(stats.chi2.sf(stat, df=1))
    else:
        p_val_mcnemar = 1.0

    # Wilcoxon signed-rank test for continuous paired SPL
    diff_spl = f_spl - a_spl
    if np.any(diff_spl != 0):
        w_res = stats.wilcoxon(diff_spl, alternative="greater")
        p_val_wilcoxon = float(w_res.pvalue)
    else:
        p_val_wilcoxon = 1.0

    return {
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p_value": p_val_mcnemar,
        "wilcoxon_p_value": p_val_wilcoxon,
        "sr_diff": float(np.mean(f_sr) - np.mean(a_sr)),
        "spl_diff": float(np.mean(f_spl) - np.mean(a_spl)),
        "bootstrap_sr": boot_sr,
        "bootstrap_spl": boot_spl,
    }

