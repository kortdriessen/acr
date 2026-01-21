import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.power import TTestPower

import acr.utils


def calculate_wilcoxon_z(W_statistic, N, ties=None, continuity_correction=True):
    """
    Computes the Z-statistic for a Wilcoxon signed-rank test.

    Parameters:
    - W_statistic (float): The sum of ranks for one group (e.g., sum of positive ranks).
                           If your W is min(W_pos, W_neg), you should use the sum of positive ranks
                           or sum of negative ranks here. For instance, if W_pos + W_neg = N(N+1)/2,
                           and your W_statistic is W_pos.
    - N (int): The number of pairs with non-zero differences.
    - ties (list of int, optional): A list where each element is the count of
                                    observations in a group of tied absolute ranks.
                                    Example: if there are two differences tied for one rank,
                                    and three differences tied for another rank, ties = [2, 3].
                                    Defaults to None (no ties).
    - continuity_correction (bool): Whether to apply the continuity correction.
                                    Defaults to True.

    Returns:
    - float: The calculated Z-statistic.
    - None: If N is too small or sigma_W is zero.
    """

    if N <= 0:
        print("N must be greater than 0.")
        return None

    mu_W = N * (N + 1) / 4

    # Calculate variance
    variance_W_no_ties = N * (N + 1) * (2 * N + 1) / 24

    tie_correction_term = 0
    if ties:
        for t_j in ties:
            tie_correction_term += t_j**3 - t_j

    variance_W = variance_W_no_ties - (tie_correction_term / 48)

    if variance_W <= 0:  # Avoid division by zero or sqrt of negative
        print(
            "Variance is zero or negative. Cannot compute Z-statistic (check N and ties)."
        )
        return None

    sigma_W = math.sqrt(variance_W)

    if sigma_W == 0:
        print("Standard deviation (sigma_W) is zero. Cannot compute Z-statistic.")
        return None
    # Calculate Z-statistic with continuity correction
    if continuity_correction:
        if W_statistic > mu_W:
            z_val = (W_statistic - mu_W - 0.5) / sigma_W
        elif W_statistic < mu_W:
            z_val = (W_statistic - mu_W + 0.5) / sigma_W
        else:  # W_statistic == mu_W
            z_val = 0.0
    else:
        z_val = (W_statistic - mu_W) / sigma_W

    return z_val


def calculate_wilx_r(W_statistic, N, ties=None, continuity_correction=True):
    z = calculate_wilcoxon_z(W_statistic, N, ties, continuity_correction)
    return np.abs(z) / np.sqrt(N)


def write_stats_result(
    test_name,
    test_type,
    test_statistic,
    p_value,
    effect_size_method,
    effect_size,
    notes="",
    review=False,
):
    if review:
        stat_path = os.path.join(acr.utils.materials_root, "stats_summary_review.xlsx")
    else:
        stat_path = os.path.join(acr.utils.materials_root, "stats_summary.xlsx")

    # convert any array-like effect_size to a string for Excel compatibility
    if isinstance(effect_size, (np.ndarray, list, tuple)):
        effect_size = ",".join(map(str, effect_size))

    # if the file exists, load it; otherwise start a fresh DataFrame
    if os.path.exists(stat_path):
        df = pd.read_excel(stat_path)
    else:
        df = pd.DataFrame(
            columns=[
                "test_name",
                "test_type",
                "test_statistic",
                "p_value",
                "effect_size_method",
                "effect_size",
                "notes",
            ]
        )

    # if test_name already exists, update the row
    if test_name in df["test_name"].values:
        df.loc[df["test_name"] == test_name, "test_type"] = test_type
        df.loc[df["test_name"] == test_name, "test_statistic"] = test_statistic
        df.loc[df["test_name"] == test_name, "p_value"] = p_value
        df.loc[df["test_name"] == test_name, "effect_size_method"] = effect_size_method
        df.loc[df["test_name"] == test_name, "effect_size"] = effect_size
        df.loc[df["test_name"] == test_name, "notes"] = notes

    else:
        # build the new row
        new_row = {
            "test_name": test_name,
            "test_type": test_type,
            "test_statistic": test_statistic,
            "p_value": p_value,
            "effect_size_method": effect_size_method,
            "effect_size": effect_size,
            "notes": notes,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # save
    df.to_excel(stat_path, index=False)


def compute_power(effect_size, n, alpha=0.05):
    # 3) Power for a paired/one-sample t-test with n = 6
    power_model = TTestPower()

    power_n = power_model.solve_power(
        effect_size=abs(effect_size),  # use magnitude
        nobs=n,
        alpha=alpha,
        alternative="two-sided",
    )

    # (Optional) How many animals needed for 80% or 90% power?
    n_for_80 = power_model.solve_power(
        effect_size=abs(effect_size),
        nobs=None,
        alpha=alpha,
        power=0.8,
        alternative="two-sided",
    )
    n_for_90 = power_model.solve_power(
        effect_size=abs(effect_size),
        nobs=None,
        alpha=alpha,
        power=0.9,
        alternative="two-sided",
    )
    return power_n, n_for_80, n_for_90


@dataclass
class GroupSummary:
    n: int
    mean: float
    sd: float
    se: float


def summarize_group(deltas: np.ndarray) -> GroupSummary:
    """
    Summarize per-mouse effects (deltas) for a group.

    Parameters
    ----------
    deltas : array-like
        1D array of per-mouse effects, e.g. SWA_optrode - SWA_contra
        during the rebound period.

    Returns
    -------
    GroupSummary
        n, mean, sample SD, and SE of the mean.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1:
        raise ValueError("deltas must be a 1D array of per-mouse effects.")

    n = deltas.size
    mean = deltas.mean()
    sd = deltas.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan

    return GroupSummary(n=n, mean=mean, sd=sd, se=se)


@dataclass
class ReplicationBFResult:
    bf_0r: float  # BF in favor of H0 (no effect) vs Hr (OFF-like effect)
    bf_r0: float  # BF in favor of Hr vs H0 (replication BF)
    mu0: float  # prior mean for HALO effect (from OFF)
    tau: float  # prior SD for HALO effect (from OFF)
    mu_hat_halo: float  # observed HALO group mean
    se_halo: float  # SE of HALO group mean
    mu_post: float  # posterior mean under Hr, after HALO
    sd_post: float  # posterior SD under Hr, after HALO


def replication_bayes_factor(
    deltas_off: np.ndarray,
    deltas_halo: np.ndarray,
) -> ReplicationBFResult:
    """
    Compute a simple replication Bayes factor for HALO vs OFF-induction effect.

    OFF-induction (ACR+SOM) deltas define the "replication prior":
    mu ~ Normal(mu0, tau^2), with (mu0, tau) taken from the OFF group
    posterior (approximated as Normal(mean_off, SE_off^2)).

    The HALO group provides data: mean_halo ± SE_halo.

    We compute:

        BF_0r = p(mean_halo | H0: mu=0) / p(mean_halo | Hr: mu~Normal(mu0, tau^2))

    Under H0, mean_halo ~ Normal(0, SE_halo^2).
    Under Hr, mean_halo ~ Normal(mu0, SE_halo^2 + tau^2).

    Parameters
    ----------
    deltas_off : array-like
        Per-mouse effects (optrode - contra) for OFF-induction mice
        (e.g. concatenate ACR and SOM).
    deltas_halo : array-like
        Per-mouse effects (optrode - contra) for HALO mice.

    Returns
    -------
    ReplicationBFResult
        Contains BF_0r, BF_r0, the prior and posterior parameters.
    """
    # 1) Summarize OFF-induction and HALO groups
    off = summarize_group(deltas_off)
    halo = summarize_group(deltas_halo)

    if halo.n < 1 or off.n < 1:
        raise ValueError("Both groups must have n >= 1.")

    if halo.se == 0 or np.isnan(halo.se):
        raise ValueError("HALO SE is zero/NaN; need variability for this method.")

    if off.se == 0 or np.isnan(off.se):
        raise ValueError("OFF SE is zero/NaN; need variability for prior.")

    # 2) Define prior for HALO effect under the replication hypothesis Hr
    mu0 = off.mean  # prior mean (OFF effect)
    tau = off.se  # prior SD (posterior SD of OFF mean)

    mu_hat_halo = halo.mean
    s = halo.se  # SE of HALO mean

    # 3) Marginal distribution of HALO mean under H0 and Hr
    # H0: mean_halo ~ Normal(0, s^2)
    # Hr: mean_halo ~ Normal(mu0, s^2 + tau^2)
    s0 = s
    sr = np.sqrt(s**2 + tau**2)

    # Evaluate densities at observed mean
    p_mean_given_H0 = norm.pdf(mu_hat_halo, loc=0.0, scale=s0)
    p_mean_given_Hr = norm.pdf(mu_hat_halo, loc=mu0, scale=sr)

    bf_0r = p_mean_given_H0 / p_mean_given_Hr
    bf_r0 = 1.0 / bf_0r

    # 4) Posterior for mu under Hr after observing HALO mean
    # Normal-Normal conjugacy:
    # posterior variance:
    #   1/sigma_post^2 = 1/tau^2 + 1/s^2
    # posterior mean:
    #   mu_post = sigma_post^2 * (mu0/tau^2 + mu_hat_halo/s^2)
    sigma_post_sq = 1.0 / (1.0 / tau**2 + 1.0 / s**2)
    sigma_post = np.sqrt(sigma_post_sq)
    mu_post = sigma_post_sq * (mu0 / tau**2 + mu_hat_halo / s**2)

    return ReplicationBFResult(
        bf_0r=bf_0r,
        bf_r0=bf_r0,
        mu0=mu0,
        tau=tau,
        mu_hat_halo=mu_hat_halo,
        se_halo=s,
        mu_post=mu_post,
        sd_post=sigma_post,
    )


def compute_sem(
    x: np.ndarray, *, axis: int = 0, ddof: int = 1, nan_policy: str = "propagate"
) -> np.ndarray:
    """
    Compute SEM across trials for an array shaped (trials, timepoints) by default.

    Parameters
    ----------
    x : np.ndarray
        Input data. Typically shape (trials, timepoints).
    axis : int
        Axis to treat as trials (default 0).
    ddof : int
        Delta degrees of freedom for the std (default 1, i.e. sample std).
    nan_policy : {"propagate", "omit"}
        - "propagate": NaNs propagate (uses np.std, counts all entries).
        - "omit": ignore NaNs (uses np.nanstd, counts non-NaNs).

    Returns
    -------
    np.ndarray
        SEM values with the trial axis removed (e.g., shape (timepoints,)).
    """
    x = np.asarray(x)

    if nan_policy == "omit":
        n = np.sum(~np.isnan(x), axis=axis)
        sd = np.nanstd(x, axis=axis, ddof=ddof)
    elif nan_policy == "propagate":
        n = x.shape[axis]
        sd = np.std(x, axis=axis, ddof=ddof)
    else:
        raise ValueError("nan_policy must be 'propagate' or 'omit'")

    with np.errstate(invalid="ignore", divide="ignore"):
        return sd / np.sqrt(n)


def get_positions_and_values_df(array, value_name, subject):
    # Get non-nan values and their positions from array array
    dfs = []
    mask = ~np.isnan(array)
    positions = np.argwhere(mask)  # Returns array of [row, col] pairs
    values = array[mask]  # Flattened array of non-nan values
    # Convert to lists
    position_list = positions.tolist()  # List of [axis-0, axis-1] positions
    value_list = values.tolist()  # List of corresponding values
    ix = 0
    for pos, val in zip(position_list, value_list):
        i = pos[0]
        j = pos[1]
        df = pd.DataFrame(
            {"unit_i": i, "unit_j": j, value_name: val, "mouse": subject}, index=[ix]
        )
        ix += 1
        dfs.append(df)
    return pd.concat(dfs)


def fisher_z(r, eps=1e-6):
    r = np.asarray(r, dtype=float)
    r = np.clip(r, -1 + eps, 1 - eps)
    return np.arctanh(r)


def pooled_dyadic_probe_effect_pymc(
    df: pd.DataFrame,
    probe_col: str = "probe",
    contra_label: str = "contra",
    optrode_label: str = "optrode",
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 0,
):
    """
    Pooled dyadic mixed model testing whether (rebound - lateBL) differs between probes.
    Returns a single global inference for beta (optrode vs contra) using all pairs while
    accounting for mouse clustering + dyadic dependence (shared neurons).
    """
    df = df.copy()

    # Outcome: paired difference in Fisher-z space
    df["dz"] = fisher_z(df["sttc_rebound"].to_numpy()) - fisher_z(
        df["sttc_late"].to_numpy()
    )
    y = df["dz"].to_numpy().astype(float)

    # Probe indicator: 0 = contra, 1 = optrode
    probe = df[probe_col].astype(str).to_numpy()
    is_optrode = (probe == optrode_label).astype(int)
    if not np.all((probe == contra_label) | (probe == optrode_label)):
        bad = sorted(set(probe) - {contra_label, optrode_label})
        raise ValueError(f"Unexpected probe labels: {bad}")

    # Mouse index
    mice = pd.Index(df["mouse"].astype("category").cat.categories)
    mouse_idx = df["mouse"].astype("category").cat.codes.to_numpy()
    n_mice = len(mice)

    # Unit indices: make units unique within mouse×probe
    u1 = (
        df["mouse"].astype(str)
        + "||"
        + df[probe_col].astype(str)
        + "||"
        + df["unit_i"].astype(str)
    )
    u2 = (
        df["mouse"].astype(str)
        + "||"
        + df[probe_col].astype(str)
        + "||"
        + df["unit_j"].astype(str)
    )
    all_units = pd.Index(pd.unique(pd.concat([u1, u2], ignore_index=True)))
    unit_map = {u: k for k, u in enumerate(all_units)}
    i_idx = u1.map(unit_map).to_numpy()
    j_idx = u2.map(unit_map).to_numpy()
    n_units = len(all_units)

    import pymc as pm

    with pm.Model() as model:
        # Global mean and probe effect
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=1.0)  # optrode vs contra on dz

        # Mouse random intercept + random slope for probe (within-mouse pairing)
        sigma_mouse = pm.HalfNormal("sigma_mouse", sigma=1.0)
        z_mouse = pm.Normal("z_mouse", 0.0, 1.0, shape=n_mice)
        u_mouse = sigma_mouse * z_mouse

        sigma_slope = pm.HalfNormal("sigma_slope", sigma=1.0)
        z_slope = pm.Normal("z_slope", 0.0, 1.0, shape=n_mice)
        s_mouse = sigma_slope * z_slope

        # Dyadic unit random effects (multi-membership)
        sigma_unit = pm.HalfNormal("sigma_unit", sigma=1.0)
        z_unit = pm.Normal("z_unit", 0.0, 1.0, shape=n_units)
        u_unit = sigma_unit * z_unit

        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu_hat = (
            alpha
            + beta * is_optrode
            + u_mouse[mouse_idx]
            + s_mouse[mouse_idx] * is_optrode
            + u_unit[i_idx]
            + u_unit[j_idx]
        )

        pm.Normal("obs", mu=mu_hat, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )

    beta_samps = idata.posterior["beta"].values.reshape(-1)
    prob_beta_lt0 = float(
        np.mean(beta_samps < 0)
    )  # evidence optrode lowers rebound synchrony increase
    p_like_two_sided = 2.0 * min(prob_beta_lt0, 1.0 - prob_beta_lt0)

    out = {
        "n_pairs": int(len(df)),
        "n_mice": int(n_mice),
        "n_units": int(n_units),
        "beta_mean": float(beta_samps.mean()),
        "beta_ci95": (
            float(np.quantile(beta_samps, 0.025)),
            float(np.quantile(beta_samps, 0.975)),
        ),
        "P(beta<0)": prob_beta_lt0,
        "p_two_sided_like": float(p_like_two_sided),
    }
    return out, idata


def pooled_dyadic_test_cond_pymc(
    df: pd.DataFrame,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 0,
):
    """
    Pooled single test for late-baseline vs rebound STTC using a dyadic mixed model:
        dz = atanh(STTC_rebound) - atanh(STTC_late)
        dz ~ mu + mouse_RE + unit_RE(i) + unit_RE(j) + noise

    Returns:
        summary dict with posterior mean/CI for mu and a two-sided p-like value:
            p_two_sided = 2 * min(P(mu>0), P(mu<0))
    """
    # --- build dz (paired difference) ---
    df = df.copy()
    dz = fisher_z(df["sttc_rebound"].to_numpy()) - fisher_z(df["sttc_late"].to_numpy())
    df["dz"] = dz

    # --- index mice ---
    mice = pd.Index(df["mouse"].astype("category").cat.categories)
    mouse_idx = df["mouse"].astype("category").cat.codes.to_numpy()
    n_mice = len(mice)

    # --- index units (unique per mouse) ---
    # make globally unique unit ids by combining mouse + unit label
    u1 = df["mouse"].astype(str) + "||" + df["unit_i"].astype(str)
    u2 = df["mouse"].astype(str) + "||" + df["unit_j"].astype(str)
    all_units = pd.Index(pd.unique(pd.concat([u1, u2], ignore_index=True)))
    unit_map = {u: k for k, u in enumerate(all_units)}
    i_idx = u1.map(unit_map).to_numpy()
    j_idx = u2.map(unit_map).to_numpy()
    n_units = len(all_units)

    y = df["dz"].to_numpy().astype(float)

    # --- dyadic mixed model in PyMC ---
    import pymc as pm

    with pm.Model() as model:
        # global mean effect we care about
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)

        # mouse random intercepts
        sigma_mouse = pm.HalfNormal("sigma_mouse", sigma=1.0)
        z_mouse = pm.Normal("z_mouse", mu=0.0, sigma=1.0, shape=n_mice)
        u_mouse = sigma_mouse * z_mouse

        # unit random effects (multi-membership: add for i and j)
        sigma_unit = pm.HalfNormal("sigma_unit", sigma=1.0)
        z_unit = pm.Normal("z_unit", mu=0.0, sigma=1.0, shape=n_units)
        u_unit = sigma_unit * z_unit

        # observation noise
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        mu_hat = mu + u_mouse[mouse_idx] + u_unit[i_idx] + u_unit[j_idx]

        pm.Normal("obs", mu=mu_hat, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
        )

    # --- summarize mu and compute a p-like two-sided tail area ---
    mu_samps = idata.posterior["mu"].values.reshape(-1)
    prob_pos = float(np.mean(mu_samps > 0))
    p_two_sided = 2.0 * min(prob_pos, 1.0 - prob_pos)

    out = {
        "n_pairs": int(len(df)),
        "n_mice": int(n_mice),
        "n_units": int(n_units),
        "mu_mean": float(np.mean(mu_samps)),
        "mu_ci95": (
            float(np.quantile(mu_samps, 0.025)),
            float(np.quantile(mu_samps, 0.975)),
        ),
        "P(mu>0)": prob_pos,
        "p_two_sided_like": float(p_two_sided),
    }
    return out, idata


import statsmodels.api as sm
from scipy import stats


def fit_line_and_mean_ci(
    x, y, x_grid=None, ci=0.95, method="parametric", n_boot=2000, seed=0
):
    """
    Fit OLS line y ~ 1 + x and return:
      x_grid, y_hat, ci_lower, ci_upper

    CI is for the *mean* E[y | x], not a prediction interval.

    method:
      - "parametric": classic OLS CI using t distribution
      - "bootstrap": bootstrap resampling of (x,y) pairs
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        raise ValueError(f"Need at least 3 points; got {x.size}.")
    if np.allclose(x, x[0]):
        raise ValueError("x is constant; cannot fit a line.")

    if x_grid is None:
        x_grid = np.linspace(x.min(), x.max(), 200)
    else:
        x_grid = np.asarray(x_grid, dtype=float).ravel()

    # Fit OLS
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()

    Xg = sm.add_constant(x_grid)
    y_hat = Xg @ res.params

    if method == "parametric":
        # Classic mean CI: y_hat +/- t * SE(y_hat)
        # SE(y_hat) = sqrt(diag(Xg * Cov(beta) * Xg^T))
        covb = res.cov_params()
        se = np.sqrt(np.sum((Xg @ covb) * Xg, axis=1))

        alpha = 1 - ci
        tcrit = stats.t.ppf(1 - alpha / 2, df=res.df_resid)
        lo = y_hat - tcrit * se
        hi = y_hat + tcrit * se

    elif method == "bootstrap":
        rng = np.random.default_rng(seed)
        n = x.size
        idx = np.arange(n)

        boot_preds = np.empty((n_boot, x_grid.size), dtype=float)
        for b in range(n_boot):
            samp = rng.choice(idx, size=n, replace=True)
            xb, yb = x[samp], y[samp]
            rb = sm.OLS(yb, sm.add_constant(xb)).fit()
            boot_preds[b] = sm.add_constant(x_grid) @ rb.params

        alpha = (1 - ci) / 2
        lo = np.quantile(boot_preds, alpha, axis=0)
        hi = np.quantile(boot_preds, 1 - alpha, axis=0)

    else:
        raise ValueError("method must be 'parametric' or 'bootstrap'.")

    return x_grid, y_hat, lo, hi
