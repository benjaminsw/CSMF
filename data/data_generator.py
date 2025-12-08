"""
Simple data generator for AMF-VI synthetic datasets.
Usage: from data.data_generator import generate_data
"""

from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
import torch

# Optional deps for auto-loading/preprocessing (you can remove if you pass X,y)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

try:
    from sklearn.datasets import fetch_openml  # for 'credit-g' and 'adult' (OpenML)
    _HAVE_OPENML = True
except Exception:
    _HAVE_OPENML = False



def create_banana_data(n_samples=1000, noise=0.1):
    """Banana: N(x2; x1^2/4, 1) * N(x1; 0, 2)"""
    x1 = np.random.normal(0, np.sqrt(2), n_samples)
    x2 = np.random.normal(x1**2 / 4, 1.0, n_samples)
    if noise > 0:
        x1 += np.random.normal(0, noise, n_samples)
        x2 += np.random.normal(0, noise, n_samples)
    return torch.tensor(np.column_stack([x1, x2]), dtype=torch.float32)


def create_x_shape_data(n_samples=1000, noise=0.1):
    """X-shape: Mixture of two diagonal Gaussians"""
    n_half = n_samples // 2
    cov1 = np.array([[2.0, 1.8], [1.8, 2.0]])
    cov2 = np.array([[2.0, -1.8], [-1.8, 2.0]])
    
    samples1 = np.random.multivariate_normal([0, 0], cov1, n_half)
    samples2 = np.random.multivariate_normal([0, 0], cov2, n_samples - n_half)
    
    data = np.vstack([samples1, samples2])
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


def create_bimodal_shared_base(n_samples=1000, separation=3.0, noise=0.3):
    """Bimodal with shared covariance"""
    n_half = n_samples // 2
    cov = np.array([[0.5, 0.0], [0.0, 0.5]])
    
    samples1 = np.random.multivariate_normal([-separation/2, 0], cov, n_half)
    samples2 = np.random.multivariate_normal([separation/2, 0], cov, n_samples - n_half)
    
    data = np.vstack([samples1, samples2])
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


def create_bimodal_different_base(n_samples=1000, separation=6, noise=0.2):
    """Bimodal with different covariances"""
    n_half = n_samples // 2
    cov1 = np.array([[0.8, 0.2], [0.2, 0.3]])
    cov2 = np.array([[0.3, -0.1], [-0.1, 0.6]])
    
    samples1 = np.random.multivariate_normal([-separation/2, -1.0], cov1, n_half)
    samples2 = np.random.multivariate_normal([separation/2, 1.0], cov2, n_samples - n_half)
    
    data = np.vstack([samples1, samples2])
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


def create_multimodal_gaussian_mixture(n_samples=1000, n_modes=3, noise=0.1):
    """Multimodal Gaussian mixture"""
    modes = [(-2.0, -1.0), (2.0, 1.0), (0.0, 2.5)][:n_modes]
    samples_per_mode = n_samples // len(modes)
    mode_var = [0.7, 1, 0.3]
    all_samples = []
    count = 0
    for mode in modes:
        samples = np.random.multivariate_normal(mode, mode_var[count] * np.eye(2), samples_per_mode)
        all_samples.append(samples)
        count += 1
    
    data = np.vstack(all_samples)
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


import numpy as np
import torch

# Optional deps for auto-loading/preprocessing (remove if you pass X,y directly)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

try:
    from sklearn.datasets import fetch_openml  # for 'credit-g' and 'adult'
    _HAVE_OPENML = True
except Exception:
    _HAVE_OPENML = False


def _log_likelihood_blr(w, X2, y, b0):
    """Logistic regression log-likelihood (no prior) for 2D weights."""
    z = b0 + X2 @ w
    # stable log-sigmoid:
    # log(sigmoid(z)) = -softplus(-z); log(1-sigmoid(z)) = -(z + softplus(-z))
    ll_pos = -np.log1p(np.exp(-z))           # y=1
    ll_neg = -(z + np.log1p(np.exp(-z)))     # y=0
    return np.sum(np.where(y == 1, ll_pos, ll_neg))


def _ess_step(w_cur, prior_chol, loglike_fn, rng):
    """One Elliptical Slice Sampling step for w with Gaussian prior N(0, prior_chol prior_chol^T)."""
    v = prior_chol @ rng.normal(size=w_cur.shape)  # prior draw
    logy = loglike_fn(w_cur) + np.log(rng.uniform())
    theta = rng.uniform(0, 2 * np.pi)
    theta_min = theta - 2 * np.pi
    theta_max = theta
    while True:
        w_prop = w_cur * np.cos(theta) + v * np.sin(theta)
        if loglike_fn(w_prop) >= logy:
            return w_prop
        if theta < 0:
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)


def _prepare_dataset_2d(dataset="german_credit", random_state=0):
    """Returns (X2, y, b0) with 2D PCA features and intercept b0=logit(class prior)."""
    rng = check_random_state(random_state)
    if not _HAVE_OPENML:
        raise RuntimeError("fetch_openml unavailable. Pass X,y to create_blr_posterior_2d instead.")

    import pandas as pd  # local import to avoid hard dep if not used

    if dataset.lower() in ["german_credit", "german", "credit-g"]:
        dset = fetch_openml(name="credit-g", version=1, as_frame=True)
        df = dset.frame
        y = (df["class"].to_numpy() == "good").astype(int)
        X = df.drop(columns=["class"])
        X = pd.get_dummies(X, drop_first=True)
    elif dataset.lower() in ["adult", "census", "adult-census"]:
        dset = fetch_openml(name="adult", version=2, as_frame=True)
        df = dset.frame
        y = (df["class"].to_numpy() == ">50K").astype(int)
        X = df.drop(columns=["class"])
        X = pd.get_dummies(X, drop_first=True)
    else:
        raise ValueError("dataset must be 'german_credit' or 'adult'")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    X = StandardScaler().fit_transform(X)
    X2 = PCA(n_components=2, random_state=random_state).fit_transform(X)

    p = y.mean()
    p = np.clip(p, 1e-4, 1 - 1e-4)
    b0 = float(np.log(p / (1 - p)))
    return X2, y.astype(int), b0


def create_blr_posterior_2d(
    n_samples=4000,
    dataset="german_credit",
    X=None,
    y=None,
    random_state=0,
    prior_type="studentt",   # "gaussian" or "studentt"
    prior_var=5.0,          # base variance (per-dim) of Σ0 = prior_var * I
    nu=5.0,                  # Student-t degrees of freedom (heavier tails if smaller; keep > 2)
    n_warmup=2000,
    n_thin=2
):
    """
    Draw posterior samples of 2D logistic-regression weights using ESS.
    Supports Gaussian prior (original) and Student-t prior via scale-mixture (Gamma) augmentation.

    Returns: torch.FloatTensor of shape (n_samples, 2).
    """
    rng = np.random.default_rng(random_state)

    # Data prep: either load+PCA->2D or use provided X,y then PCA->2D
    if X is None or y is None:
        X2, y_bin, b0 = _prepare_dataset_2d(dataset=dataset, random_state=random_state)
    else:
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float64)
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        y_bin = (y_np > 0).astype(int)
        X_np = StandardScaler().fit_transform(X_np)
        X2 = PCA(n_components=2, random_state=random_state).fit_transform(X_np)
        p = y_bin.mean()
        p = np.clip(p, 1e-4, 1 - 1e-4)
        b0 = float(np.log(p / (1 - p)))

    d = 2
    Sigma0 = prior_var * np.eye(d)
    Sigma0_inv = (1.0 / prior_var) * np.eye(d)

    # Initialize w and (optionally) lambda for Student-t
    if prior_type.lower() == "gaussian":
        w = rng.normal(scale=np.sqrt(prior_var), size=d)
        lambda_prec = 1.0  # not used
    elif prior_type.lower() == "studentt":
        # lambda ~ Gamma(nu/2, rate=nu/2). numpy uses scale=1/rate.
        lambda_prec = rng.gamma(shape=nu / 2.0, scale=2.0 / nu)
        w = rng.normal(scale=np.sqrt(prior_var / lambda_prec), size=d)
    else:
        raise ValueError("prior_type must be 'gaussian' or 'studentt'")

    def logpost_wo_prior(w_vec):
        return _log_likelihood_blr(w_vec, X2, y_bin, b0)

    def _sample_lambda_given_w(w_vec):
        # p(lambda | w) = Gamma(shape=(nu+d)/2, rate=(nu + w^T Σ0^{-1} w)/2)
        quad = float(w_vec @ (Sigma0_inv @ w_vec))
        shape = 0.5 * (nu + d)
        rate = 0.5 * (nu + quad)
        scale = 1.0 / rate
        return rng.gamma(shape=shape, scale=scale)

    def _prior_chol(lambda_prec_val):
        # chol of Σ = Σ0 / lambda
        s = np.sqrt(prior_var / lambda_prec_val)
        return s * np.eye(d)

    # Warmup
    for _ in range(int(n_warmup)):
        if prior_type.lower() == "studentt":
            lambda_prec = _sample_lambda_given_w(w)
            prior_chol = _prior_chol(lambda_prec)
        else:  # gaussian
            prior_chol = np.sqrt(prior_var) * np.eye(d)
        w = _ess_step(w, prior_chol, logpost_wo_prior, rng)

    # Draws
    samples = []
    total_needed = int(n_samples) * int(max(1, n_thin))
    for i in range(total_needed):
        if prior_type.lower() == "studentt":
            lambda_prec = _sample_lambda_given_w(w)
            prior_chol = _prior_chol(lambda_prec)
        else:
            prior_chol = np.sqrt(prior_var) * np.eye(d)

        w = _ess_step(w, prior_chol, logpost_wo_prior, rng)

        if (i + 1) % n_thin == 0:
            samples.append(w.copy())

    data = np.vstack(samples)
    return torch.tensor(data, dtype=torch.float32)

def create_two_moons_data(n_samples=1000, noise=0.1):
    """Two moons from sklearn"""
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    return torch.tensor(X, dtype=torch.float32)


def create_concentric_rings(n_samples=1000, noise=0.1):
    """Concentric rings"""
    n_half = n_samples // 2
    
    # Inner ring
    theta1 = np.random.uniform(0, 2*np.pi, n_half)
    r1 = np.random.normal(1.0, noise, n_half)
    x1, y1 = r1 * np.cos(theta1), r1 * np.sin(theta1)
    
    # Outer ring
    theta2 = np.random.uniform(0, 2*np.pi, n_samples - n_half)
    r2 = np.random.normal(2.5, noise, n_samples - n_half)
    x2, y2 = r2 * np.cos(theta2), r2 * np.sin(theta2)
    
    data = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)

import numpy as np
import torch

# Optional deps for auto-loading/preprocessing (remove if you pass X,y)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

try:
    from sklearn.datasets import fetch_openml
    _HAVE_OPENML = True
except Exception:
    _HAVE_OPENML = False


# --------------------------
# Utilities
# --------------------------
def _prepare_dataset_2d_probit(dataset="german_credit", random_state=0):
    """
    Returns (X2, y, b0) where:
      - X2: (n, 2) PCA-reduced, standardized features,
      - y : {0,1} labels,
      - b0: fixed intercept = Phi^{-1}(mean(y)) for probit.
    """
    rng = check_random_state(random_state)
    if not _HAVE_OPENML:
        raise RuntimeError(
            "fetch_openml unavailable. Pass X,y directly to create_probit_posterior_2d(...)."
        )

    import pandas as pd  # local import

    if dataset.lower() in ["german_credit", "german", "credit-g"]:
        dset = fetch_openml(name="credit-g", version=1, as_frame=True)
        df = dset.frame
        y = (df["class"].to_numpy() == "good").astype(int)
        X = pd.get_dummies(df.drop(columns=["class"]), drop_first=True)
    elif dataset.lower() in ["adult", "census", "adult-census"]:
        dset = fetch_openml(name="adult", version=2, as_frame=True)
        df = dset.frame
        y = (df["class"].to_numpy() == ">50K").astype(int)
        X = pd.get_dummies(df.drop(columns=["class"]), drop_first=True)
    else:
        raise ValueError("dataset must be 'german_credit' or 'adult'")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    X = StandardScaler().fit_transform(X)
    X2 = PCA(n_components=2, random_state=random_state).fit_transform(X)

    # Probit intercept fixed to class prior: b0 = Phi^{-1}(p)
    p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    # use torch erfinv for a robust Phi^{-1}:
    p_t = torch.tensor(p, dtype=torch.float64)
    b0 = float(np.sqrt(2.0) * torch.erfinv(2.0 * p_t - 1.0))

    return X2, y.astype(int), b0


def _trunc_norm_sample(mu_vec, lower=None, upper=None, eps=1e-12):
    """
    Vectorized sampling from N(mu, 1) truncated to (lower, upper).
    Uses inverse-CDF with erf/erfinv (torch). lower/upper can be scalars or None.
    """
    mu = torch.as_tensor(mu_vec, dtype=torch.float64)
    N = mu.numel()

    # Standard normal CDF and PPF via erf/erfinv
    def Phi(x):  # x: tensor
        return 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

    def Phi_inv(p):  # p in (0,1)
        p = torch.clamp(p, eps, 1.0 - eps)
        return np.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)

    if lower is not None and upper is not None:
        a = (lower - mu)
        b = (upper - mu)
        u = torch.rand(N, dtype=torch.float64)
        lo = Phi(a)
        hi = Phi(b)
        u = lo + (hi - lo) * u
        return mu + Phi_inv(u)

    if lower is not None:
        a = (lower - mu)
        lo = Phi(a)
        u = lo + (1.0 - lo) * torch.rand(N, dtype=torch.float64)
        return mu + Phi_inv(u)

    if upper is not None:
        b = (upper - mu)
        hi = Phi(b)
        u = torch.rand(N, dtype=torch.float64) * hi
        return mu + Phi_inv(u)

    # untruncated (shouldn't be used here)
    return mu + torch.randn(N, dtype=torch.float64)


# --------------------------
# Main generator
# --------------------------
@torch.no_grad()
def create_probit_posterior_2d(
    n_samples=10000,
    dataset="german_credit",
    X=None,
    y=None,
    random_state=0,
    prior_var=30.0,        # prior variance on weights (N(0, prior_var * I2))
    burn_in=1500,
    thin=5,
):
    """
    Returns posterior samples for 2D probit regression weights using Albert–Chib Gibbs sampling.
    Model:
      z_i = b0 + x_i^T w + eps_i,  eps_i ~ N(0,1)
      y_i = 1{ z_i > 0 }
      w ~ N(0, prior_var * I_2),  b0 fixed to Phi^{-1}(mean(y)) to keep 2D parameter space.
    Output:
      torch.FloatTensor of shape (n_samples, 2)
    """
    rng = np.random.default_rng(random_state)

    # --- Data prep: PCA→2D, standardize; get fixed intercept b0 ---
    if X is None or y is None:
        X2, y_bin, b0 = _prepare_dataset_2d_probit(dataset=dataset, random_state=random_state)
    else:
        X_np = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float64)
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        y_bin = (y_np > 0).astype(int)

        X_np = StandardScaler().fit_transform(X_np)
        X2 = PCA(n_components=2, random_state=random_state).fit_transform(X_np)

        p = float(np.clip(y_bin.mean(), 1e-6, 1 - 1e-6))
        b0 = float(np.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0, dtype=torch.float64)))

    # cast to double for stability
    X2 = np.asarray(X2, dtype=np.float64)
    y_bin = y_bin.astype(int)
    N, d = X2.shape
    assert d == 2, "PCA must reduce to 2D."

    # Precompute for Gaussian update
    XtX = X2.T @ X2                       # (2,2)
    prior_prec = (1.0 / prior_var) * np.eye(2)
    post_prec = XtX + prior_prec          # (2,2)
    post_cov  = np.linalg.inv(post_prec)  # (2,2)
    chol_cov  = np.linalg.cholesky(post_cov + 1e-12 * np.eye(2))

    # Initialize
    w = np.zeros(2, dtype=np.float64)
    z = np.where(y_bin == 1, 1.0, -1.0)  # any feasible init consistent with y

    total_iters = int(burn_in) + int(n_samples) * max(1, int(thin))
    samples = []

    for it in range(total_iters):
        # 1) Sample latent z_i | w, y: truncated normal
        mu = b0 + X2 @ w  # (N,)
        mu_t = torch.from_numpy(mu)

        # y=1 → z > 0 ; y=0 → z < 0
        idx_pos = (y_bin == 1)
        idx_neg = ~idx_pos

        z_new = np.empty(N, dtype=np.float64)

        if idx_pos.any():
            z_pos = _trunc_norm_sample(mu_t[idx_pos], lower=0.0, upper=None)
            z_new[idx_pos] = z_pos.numpy()
        if idx_neg.any():
            z_neg = _trunc_norm_sample(mu_t[idx_neg], lower=None, upper=0.0)
            z_new[idx_neg] = z_neg.numpy()

        z = z_new

        # 2) Sample w | z: Gaussian
        # mean = post_cov @ X^T (z - b0)
        rhs = X2.T @ (z - b0)             # (2,)
        mean = post_cov @ rhs             # (2,)
        w = mean + chol_cov @ rng.normal(size=2)

        # 3) Save after burn-in with thinning
        if it >= burn_in and ((it - burn_in) % thin == 0):
            samples.append(w.copy())

    arr = np.array(samples, dtype=np.float32)  # (n_samples, 2)
    return torch.from_numpy(arr)

import numpy as np
import torch

# --------------------------
# Elliptical Slice Sampling
# --------------------------
def _ess_step(u_cur, prior_chol, loglike_fn, rng):
    """
    One ESS step for zero-mean Gaussian prior with Cholesky prior_chol.
    We pass a centered state z = u - m into this function in the wrapper below.
    """
    v = prior_chol @ rng.normal(size=u_cur.shape)          # prior draw
    logy = loglike_fn(u_cur) + np.log(rng.uniform())       # vertical slice
    theta = rng.uniform(0.0, 2.0*np.pi)
    theta_min, theta_max = theta - 2.0*np.pi, theta
    while True:
        u_prop = u_cur * np.cos(theta) + v * np.sin(theta)
        if loglike_fn(u_prop) >= logy:
            return u_prop
        if theta < 0.0:
            theta_min = theta
        else:
            theta_max = theta
        theta = rng.uniform(theta_min, theta_max)

# --------------------------
# Weibull posterior generator
# --------------------------

import numpy as np
import torch

def simulate_weibull_durations(
    n=4000,
    shape=1.6,          # k > 0
    scale=2.0,          # λ > 0
    seed=0
) -> torch.Tensor:
    """
    Draw i.i.d. durations Y ~ Weibull(k=shape, λ=scale).
    NumPy's weibull(a) uses scale=1; we multiply by λ.
    Returns: torch.FloatTensor (n,)
    """
    rng = np.random.default_rng(seed)
    y = scale * rng.weibull(shape, size=n).astype(np.float64)
    y = y[(y > 0) & np.isfinite(y)]
    return torch.tensor(y, dtype=torch.float32)


def simulate_weibull_mixture_durations(
    n=4000,
    shapes=(1.2, 2.5),   # k1, k2
    scales=(1.0, 3.0),   # λ1, λ2
    mix=0.5,             # P(component=1)
    seed=0
) -> torch.Tensor:
    """
    Two-component Weibull mixture for a tougher, more skewed real-like set.
    Returns: torch.FloatTensor (n,)
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(size=n) < mix
    y = np.empty(n, dtype=np.float64)
    y[z]  = scales[0] * rng.weibull(shapes[0], size=z.sum())
    y[~z] = scales[1] * rng.weibull(shapes[1], size=(~z).sum())
    y = y[(y > 0) & np.isfinite(y)]
    return torch.tensor(y, dtype=torch.float32)


@torch.no_grad()
def create_weibull_duration_posterior_2d(
    n_samples=4000,
    burn_in=1500,
    thin=2,
    random_state=0,
    # Gaussian prior on u = [log(lambda), log(k)]
    prior_mean=(None, np.log(1.5)),   # default log(lambda) from data if None; log(k) prior mean ~ ln(1.5)
    prior_std=(0.8, 0.6),             # prior stds in log-space (tune 0.4–1.0 as needed)
    exp_clip=60.0                     # clip for exp to avoid overflow in likelihood
):
    """
    Real-data time-to-event posterior for Weibull(shape=k, scale=lambda).
    Likelihood:  y_i ~ Weibull(k, lambda),  y_i>0.
    log p(y | lambda,k) = N*(log k - k log lambda) + (k-1)*sum log y - sum (y/lambda)^k
    Parameterization: u = [log lambda, log k] ∈ R^2. Returns samples of [log lambda, log k].

    Output: torch.FloatTensor of shape (n_samples, 2)
    """
    # --- data prep ---
    y_tensor = simulate_weibull_durations(n=5000, shape=1.7, scale=2.5, seed=42)
    y = y_tensor.detach().cpu().double().clamp_min(1e-12).numpy().ravel()
    if not np.all(y > 0.0):
        raise ValueError("All durations must be > 0.")
    N = y.shape[0]
    logy = np.log(y)
    sum_logy = float(np.sum(logy))

    # --- log-likelihood in terms of u = [u1, u2] = [log lambda, log k] ---
    # loglik(u) = N*(u2 - k*u1) + (k-1)*sum_logy - sum(exp(k*(logy - u1)))
    # where k = exp(u2)
    def loglike(u):
        u1, u2 = float(u[0]), float(u[1])
        k = np.exp(u2)
        # stable compute for (y/lambda)^k = exp(k*(logy - u1))
        exparg = k * (logy - u1)
        # clip to avoid overflow in exp; underflow is fine (just ~0 contribution)
        exparg = np.clip(exparg, -exp_clip, exp_clip)
        term = np.exp(exparg).sum()
        return N * (u2 - k * u1) + (k - 1.0) * sum_logy - term

    # --- Gaussian prior on u ---
    m1 = np.log(np.median(y)) if prior_mean[0] is None else float(prior_mean[0])  # prior center for log(lambda)
    m2 = float(prior_mean[1])
    s1, s2 = float(prior_std[0]), float(prior_std[1])

    def logprior(u):
        du1, du2 = (u[0] - m1) / s1, (u[1] - m2) / s2
        return -0.5 * (du1 * du1 + du2 * du2) - np.log(2.0 * np.pi * s1 * s2)

    # centered log-likelihood for ESS (prior is N(0, diag(s^2)) on z = u - m)
    m = np.array([m1, m2], dtype=np.float64)
    prior_chol = np.diag([s1, s2])

    def loglike_centered(z):
        return loglike(z + m)

    rng = np.random.default_rng(random_state)
    z = np.zeros(2, dtype=np.float64)  # start at prior mean (u=m → z=0)

    # --- warmup ---
    for _ in range(int(burn_in)):
        z = _ess_step(z, prior_chol, loglike_centered, rng)

    # --- draws with thinning ---
    outs = []
    total = int(n_samples) * max(1, int(thin))
    for i in range(total):
        z = _ess_step(z, prior_chol, loglike_centered, rng)
        if (i + 1) % int(thin) == 0:
            u = z + m
            outs.append(u.copy())

    arr = np.array(outs, dtype=np.float32)  # columns: [log lambda, log k]
    return torch.from_numpy(arr)


def create_multimodal_gaussian_mixture5(n_samples=1000, n_modes=5, noise=0.1):
    """Multimodal Gaussian mixture"""
    modes = [(-0.5, -1.0), (0.5, 1.0), (2.0, 2.5), (0.5, -1.0), (-0.5, 1.0)][:n_modes]
    samples_per_mode = n_samples // len(modes)
    mode_var = [0.7, 0.3, 1, 0.7, 0.3]
    all_samples = []
    count = 0
    for mode in modes:
        samples = np.random.multivariate_normal(mode, mode_var[count] * np.eye(2), samples_per_mode)
        all_samples.append(samples)
        count += 1

    data = np.vstack(all_samples)
    np.random.shuffle(data)
    if noise > 0:
        data += np.random.normal(0, noise, data.shape)
    return torch.tensor(data, dtype=torch.float32)


# Registry for easy access
DATA_GENERATORS = {
    'banana': create_banana_data,
    'x_shape': create_x_shape_data,
    'bimodal_shared': create_bimodal_shared_base,
    'bimodal_different': create_bimodal_different_base,
    'multimodal': create_multimodal_gaussian_mixture,
    'two_moons': create_two_moons_data,
    'rings': create_concentric_rings,
    'BLR': create_blr_posterior_2d,
    "BPR": create_probit_posterior_2d,
    "Weibull": create_weibull_duration_posterior_2d,
    "multimodal-5": create_multimodal_gaussian_mixture5
}


def generate_data(dataset_name, n_samples=1000, **kwargs):
    """
    Generate synthetic data by name.
    
    Args:
        dataset_name: Name from DATA_GENERATORS keys
        n_samples: Number of samples
        **kwargs: Additional parameters for specific generators
    
    Returns:
        torch.Tensor: Generated data [n_samples, 2]
    """
    if dataset_name not in DATA_GENERATORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATA_GENERATORS.keys())}")
    
    return DATA_GENERATORS[dataset_name](n_samples=n_samples, **kwargs)


def get_available_datasets():
    """Get list of available dataset names."""
    return list(DATA_GENERATORS.keys())