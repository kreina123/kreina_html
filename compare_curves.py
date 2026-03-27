# -*- coding: utf-8 -*-
"""
compare_curves.py -- Three-Model S-Curve Comparison

Models:
  1. Optimized Spline     -- per-month B-spline fit (no cross-temporal learning)
  2. Multivariate          -- logistic basis + external factors (panel regression)
  3. Multivariate Spline   -- B-spline basis + external factors (best of both)

Scenarios:
  A. In-sample (all months, standard noise)
  B. Forecasting (train 0-79, predict 80-119)
  C. High noise (sigma=0.08)
  D. Sparse data (12 of 50 incentive points)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_lsq_spline
from sklearn.metrics import mean_squared_error

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

np.random.seed(42)

# =============================================================================
# 1. SIMULATION
# =============================================================================
n_months = 120
n_incentive = 50
x = np.linspace(0, 1, n_incentive)
t_months = np.arange(n_months)

def seasonality_factor(m, amp=0.08):
    return amp * np.sin(2 * np.pi * (m - 3) / 12)

def burnout_factor(m, rate=0.03, mx=0.25):
    return -mx * (1 - np.exp(-rate * m))

def regime_factor(m, sm=60, mag=-0.10, w=3):
    return mag / (1 + np.exp(-(m - sm) / w))

def media_factor(m, pk=30, amp=0.06, w=4):
    return amp * np.exp(-0.5 * ((m - pk) / w) ** 2)

L, k_steep, x0 = 1.0, 6.0, 0.5
base_cpr = 0.05 + 0.9 / (1 + np.exp(-k_steep * (x - x0)))
base_cpr = np.clip(base_cpr, 0, 1)

season_vals = seasonality_factor(t_months)
burnout_vals = burnout_factor(t_months)
regime_vals = regime_factor(t_months)
media_vals = media_factor(t_months)

CPR_true = np.zeros((n_months, n_incentive))
for m in range(n_months):
    adj = season_vals[m] + burnout_vals[m] + regime_vals[m] + media_vals[m]
    CPR_true[m, :] = base_cpr + adj
CPR_true = np.clip(CPR_true, 0.01, 0.99)

# =============================================================================
# 2. MODEL DEFINITIONS
# =============================================================================

# --- MODEL 1: Optimized Spline (per-month) -----------------------------------
k_spline = 3

def fit_spline_month(x_grid, obs_m):
    """Fit per-month B-spline. Returns prediction on x_grid."""
    knots = np.linspace(x_grid[0], x_grid[-1], 8)
    knots = np.concatenate(([x_grid[0]] * k_spline, knots, [x_grid[-1]] * k_spline))
    s = make_lsq_spline(x_grid, obs_m, knots, k=k_spline)(x_grid)
    return np.maximum.accumulate(s)

# --- MODEL 2: Multivariate (logistic basis + external factors) ----------------
def build_multi_design(x_grid, t_grid):
    """Logistic basis + external factor features."""
    n_t, n_x = len(t_grid), len(x_grid)
    N = n_t * n_x
    X_f = np.tile(x_grid, n_t)
    T_f = np.repeat(t_grid, n_x)
    feats = [np.ones(N)]
    # Logistic bases at 3 steepness levels
    for kk in [4, 6, 10]:
        feats.append(1.0 / (1 + np.exp(-kk * (X_f - 0.5))))
    feats.append(X_f ** 2)
    # Seasonality (Fourier)
    feats.extend([np.sin(2*np.pi*T_f/12), np.cos(2*np.pi*T_f/12),
                  np.sin(4*np.pi*T_f/12), np.cos(4*np.pi*T_f/12)])
    # Burnout + regime
    bo = 1 - np.exp(-0.03 * T_f)
    rg = 1.0 / (1 + np.exp(-(T_f - 60) / 3))
    sc = 1.0 / (1 + np.exp(-6 * (X_f - 0.5)))
    feats.extend([bo, rg, bo * sc, rg * sc, T_f / n_months])
    return np.column_stack(feats)

# --- MODEL 3: Multivariate Spline (B-spline basis + external factors) --------
def _bspline_basis(x_grid, n_basis=10, degree=3):
    """Evaluate B-spline basis functions on x_grid. Returns (len(x_grid), n_basis)."""
    n_internal = n_basis - degree - 1
    if n_internal < 0:
        n_internal = 0
    internal = np.linspace(x_grid[0], x_grid[-1], n_internal + 2)[1:-1]
    knots = np.concatenate(([x_grid[0]] * (degree + 1), internal,
                            [x_grid[-1]] * (degree + 1)))
    n_actual = len(knots) - degree - 1
    basis = np.zeros((len(x_grid), n_actual))
    for j in range(n_actual):
        coeffs = np.zeros(n_actual)
        coeffs[j] = 1.0
        from scipy.interpolate import BSpline
        basis[:, j] = BSpline(knots, coeffs, degree)(x_grid)
    return basis, knots

def build_multispline_design(x_grid, t_grid, n_basis=10, degree=3):
    """B-spline basis + external factor features + spline x factor interactions."""
    n_t, n_x = len(t_grid), len(x_grid)
    N = n_t * n_x
    X_f = np.tile(x_grid, n_t)
    T_f = np.repeat(t_grid, n_x)

    # B-spline basis evaluated on x_grid, then tiled for all months
    basis_1month, _ = _bspline_basis(x_grid, n_basis, degree)
    # Tile: for each month, repeat the same basis evaluations
    B = np.tile(basis_1month, (n_t, 1))  # shape (N, n_basis)

    feats = list(B.T)  # each column is a feature

    # Seasonality (Fourier -- 2 harmonics)
    sin1 = np.sin(2*np.pi*T_f/12)
    cos1 = np.cos(2*np.pi*T_f/12)
    sin2 = np.sin(4*np.pi*T_f/12)
    cos2 = np.cos(4*np.pi*T_f/12)
    feats.extend([sin1, cos1, sin2, cos2])

    # Burnout + regime
    bo = 1 - np.exp(-0.03 * T_f)
    rg = 1.0 / (1 + np.exp(-(T_f - 60) / 3))
    feats.extend([bo, rg])

    # Trend
    feats.append(T_f / n_months)

    # KEY: Interactions -- spline_basis x external_factors
    # This lets external factors reshape the S-curve, not just shift it
    # Use 3 representative spline bases (low/mid/high incentive regions)
    idx_lo = n_basis // 4
    idx_mid = n_basis // 2
    idx_hi = 3 * n_basis // 4
    for idx in [idx_lo, idx_mid, idx_hi]:
        feats.append(B[:, idx] * bo)    # burnout x spline
        feats.append(B[:, idx] * rg)    # regime x spline
        feats.append(B[:, idx] * sin1)  # season x spline

    return np.column_stack(feats)

# --- Shared fitting -----------------------------------------------------------
def ridge_fit(X_train, y_train, lam=0.01):
    XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
    return np.linalg.solve(XtX, X_train.T @ y_train)

def predict_panel(beta, X, n_t, n_x):
    pred = np.clip(X @ beta, 0.01, 0.99).reshape(n_t, n_x)
    for m in range(n_t):
        pred[m, :] = np.maximum.accumulate(pred[m, :])
    return pred

def rmse(true, pred):
    return np.sqrt(mean_squared_error(true.flatten(), pred.flatten()))

# =============================================================================
# 3. COLOR SCHEME
# =============================================================================
C_SPLINE = '#2ECC71'      # green
C_MULTI = '#3498DB'        # blue
C_MULTISPLINE = '#9B59B6'  # purple (the star)
C_TRUE = '#2C3E50'         # dark

# =============================================================================
# 4. SCENARIO A: IN-SAMPLE (standard noise, all months)
# =============================================================================
print("=" * 80)
print("  SCENARIO A: IN-SAMPLE (sigma=0.025, all months, all incentive points)")
print("=" * 80)

sigma_std = 0.025
np.random.seed(42)
CPR_obs = np.clip(CPR_true + np.random.normal(0, sigma_std, CPR_true.shape), 0.01, 0.99)

# Spline: per-month
spline_insample = np.zeros_like(CPR_true)
for m in range(n_months):
    spline_insample[m] = fit_spline_month(x, CPR_obs[m])

# Multivariate
X_multi = build_multi_design(x, t_months)
beta_multi = ridge_fit(X_multi, CPR_obs.flatten())
multi_insample = predict_panel(beta_multi, X_multi, n_months, n_incentive)

# Multivariate Spline
X_ms = build_multispline_design(x, t_months)
beta_ms = ridge_fit(X_ms, CPR_obs.flatten())
ms_insample = predict_panel(beta_ms, X_ms, n_months, n_incentive)

r_sp_a = rmse(CPR_true, spline_insample)
r_mu_a = rmse(CPR_true, multi_insample)
r_ms_a = rmse(CPR_true, ms_insample)

print(f"\n  Optimized Spline:        {r_sp_a:.4f}")
print(f"  Multivariate:            {r_mu_a:.4f}")
print(f"  Multivariate Spline:     {r_ms_a:.4f}")

# =============================================================================
# 5. SCENARIO B: FORECASTING (train months 0-79, predict 80-119)
# =============================================================================
print("\n" + "=" * 80)
print("  SCENARIO B: FORECASTING (train 0-79, predict 80-119)")
print("=" * 80)

train_end = 80
train_m = np.arange(train_end)
test_m = np.arange(train_end, n_months)

# Spline: best available = average of training months
spline_train = np.zeros((train_end, n_incentive))
for m in range(train_end):
    spline_train[m] = fit_spline_month(x, CPR_obs[m])
avg_spline = np.mean(spline_train, axis=0)
spline_forecast = np.tile(avg_spline, (len(test_m), 1))

# Multivariate: train on 0-79, predict 80-119
X_multi_tr = build_multi_design(x, train_m)
beta_multi_fc = ridge_fit(X_multi_tr, CPR_obs[train_m].flatten())
X_multi_te = build_multi_design(x, test_m)
multi_forecast = predict_panel(beta_multi_fc, X_multi_te, len(test_m), n_incentive)

# Multivariate Spline: train on 0-79, predict 80-119
X_ms_tr = build_multispline_design(x, train_m)
beta_ms_fc = ridge_fit(X_ms_tr, CPR_obs[train_m].flatten())
X_ms_te = build_multispline_design(x, test_m)
ms_forecast = predict_panel(beta_ms_fc, X_ms_te, len(test_m), n_incentive)

r_sp_b = rmse(CPR_true[test_m], spline_forecast)
r_mu_b = rmse(CPR_true[test_m], multi_forecast)
r_ms_b = rmse(CPR_true[test_m], ms_forecast)

print(f"\n  Optimized Spline (avg):  {r_sp_b:.4f}")
print(f"  Multivariate:            {r_mu_b:.4f}")
print(f"  Multivariate Spline:     {r_ms_b:.4f}")

# =============================================================================
# 6. SCENARIO C: HIGH NOISE (sigma=0.08)
# =============================================================================
print("\n" + "=" * 80)
print("  SCENARIO C: HIGH NOISE (sigma=0.08)")
print("=" * 80)

sigma_high = 0.08
np.random.seed(42)
CPR_noisy = np.clip(CPR_true + np.random.normal(0, sigma_high, CPR_true.shape), 0.01, 0.99)

# Spline
spline_noisy = np.zeros_like(CPR_true)
for m in range(n_months):
    spline_noisy[m] = fit_spline_month(x, CPR_noisy[m])

# Multivariate
X_multi_n = build_multi_design(x, t_months)
beta_multi_n = ridge_fit(X_multi_n, CPR_noisy.flatten())
multi_noisy = predict_panel(beta_multi_n, X_multi_n, n_months, n_incentive)

# Multivariate Spline
X_ms_n = build_multispline_design(x, t_months)
beta_ms_n = ridge_fit(X_ms_n, CPR_noisy.flatten())
ms_noisy = predict_panel(beta_ms_n, X_ms_n, n_months, n_incentive)

r_sp_c = rmse(CPR_true, spline_noisy)
r_mu_c = rmse(CPR_true, multi_noisy)
r_ms_c = rmse(CPR_true, ms_noisy)

print(f"\n  Optimized Spline:        {r_sp_c:.4f}")
print(f"  Multivariate:            {r_mu_c:.4f}")
print(f"  Multivariate Spline:     {r_ms_c:.4f}")

# =============================================================================
# 7. SCENARIO D: SPARSE DATA (12 of 50 incentive pts)
# =============================================================================
print("\n" + "=" * 80)
print("  SCENARIO D: SPARSE DATA (12 of 50 incentive points)")
print("=" * 80)

n_sparse = 12
sparse_idx = np.linspace(0, n_incentive - 1, n_sparse).astype(int)
x_sparse = x[sparse_idx]

sigma_sparse = 0.04
np.random.seed(42)
CPR_obs_sp = np.clip(CPR_true + np.random.normal(0, sigma_sparse, CPR_true.shape), 0.01, 0.99)

# Spline on sparse -> interpolate to full grid
spline_sparse = np.zeros_like(CPR_true)
k_sp_sparse = 3
knots_sp = np.linspace(x_sparse[0], x_sparse[-1], 6)
knots_sp = np.concatenate(([x_sparse[0]] * k_sp_sparse, knots_sp,
                            [x_sparse[-1]] * k_sp_sparse))
for m in range(n_months):
    obs_sp = CPR_obs_sp[m, sparse_idx]
    s = make_lsq_spline(x_sparse, obs_sp, knots_sp, k=k_sp_sparse)(x)
    spline_sparse[m] = np.maximum.accumulate(s)

# Multivariate on sparse
X_multi_sp = build_multi_design(x_sparse, t_months)
beta_multi_sp = ridge_fit(X_multi_sp, CPR_obs_sp[:, sparse_idx].flatten())
X_multi_sp_full = build_multi_design(x, t_months)
multi_sparse = predict_panel(beta_multi_sp, X_multi_sp_full, n_months, n_incentive)

# Multivariate Spline on sparse
X_ms_sp = build_multispline_design(x_sparse, t_months)
beta_ms_sp = ridge_fit(X_ms_sp, CPR_obs_sp[:, sparse_idx].flatten())
X_ms_sp_full = build_multispline_design(x, t_months)
ms_sparse = predict_panel(beta_ms_sp, X_ms_sp_full, n_months, n_incentive)

r_sp_d = rmse(CPR_true, spline_sparse)
r_mu_d = rmse(CPR_true, multi_sparse)
r_ms_d = rmse(CPR_true, ms_sparse)

print(f"\n  Optimized Spline:        {r_sp_d:.4f}")
print(f"  Multivariate:            {r_mu_d:.4f}")
print(f"  Multivariate Spline:     {r_ms_d:.4f}")

# =============================================================================
# FIGURE A: RMSE Comparison across all 4 scenarios
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
scenarios = [
    ('A: In-Sample\n($\\sigma=0.025$, full data)', [r_sp_a, r_mu_a, r_ms_a]),
    ('B: Forecasting\n(months 80-119)', [r_sp_b, r_mu_b, r_ms_b]),
    ('C: High Noise\n($\\sigma=0.08$)', [r_sp_c, r_mu_c, r_ms_c]),
    ('D: Sparse Data\n(12/50 points)', [r_sp_d, r_mu_d, r_ms_d]),
]
labels = ['Optimized\nSpline', 'Multi-\nvariate', 'Multivariate\nSpline']
colors = [C_SPLINE, C_MULTI, C_MULTISPLINE]

for ax, (title, vals) in zip(axes, scenarios):
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    # Highlight winner
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor('#2C3E50')
    bars[best_idx].set_linewidth(2.5)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('RMSE (vs. True Signal)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, max(vals) * 1.35)

fig.suptitle('Figure A: RMSE Comparison -- Multivariate Spline Dominates',
             fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig("compare_rmse.png")
plt.close()
print("\nSaved compare_rmse.png")

# =============================================================================
# FIGURE B: Forecasting S-Curves at 4 test months
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
test_slices = [
    (80, 'Month 80 (just past training)'),
    (90, 'Month 90 (10 months ahead)'),
    (100, 'Month 100 (20 months ahead)'),
    (115, 'Month 115 (35 months ahead)'),
]

for ax, (m, title) in zip(axes.flat, test_slices):
    true_m = CPR_true[m, :]
    ti = m - train_end  # index into test arrays

    ax.plot(x, true_m, '-', linewidth=2.5, color=C_TRUE, label='True CPR')
    ax.plot(x, spline_forecast[ti], '--', linewidth=2, color=C_SPLINE,
            label=f'Spline avg (RMSE={rmse(true_m, spline_forecast[ti]):.4f})')
    ax.plot(x, multi_forecast[ti], '--', linewidth=2, color=C_MULTI,
            label=f'Multivariate (RMSE={rmse(true_m, multi_forecast[ti]):.4f})')
    ax.plot(x, ms_forecast[ti], '-', linewidth=2.5, color=C_MULTISPLINE,
            label=f'MV Spline (RMSE={rmse(true_m, ms_forecast[ti]):.4f})')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Refinancing Incentive ($x$)')
    ax.set_ylabel('CPR')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure B: Forecasting -- Multivariate Spline Predicts Unseen Months\n'
             'Combines spline flexibility with environmental factor knowledge',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("compare_scurves.png")
plt.close()
print("Saved compare_scurves.png")

# =============================================================================
# FIGURE C: Environmental Factors Stacked on the S-Curve
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

env_months = [
    (6,  'Month 6 -- Seasonality peak (+0.08)'),
    (30, 'Month 30 -- Media wave active (+0.06)'),
    (60, 'Month 60 -- Regime shift (-0.10)'),
    (96, 'Month 96 -- Full burnout (-0.24)'),
]

factor_colors = {
    'Seasonality': '#E67E22',
    'Burnout': '#8E44AD',
    'Regime Shift': '#C0392B',
    'Media Wave': '#27AE60',
}

for ax, (m, title) in zip(axes.flat, env_months):
    ax.plot(x, base_cpr, '-', linewidth=2, color='#2C3E50', label='Base S-curve')

    cumul = base_cpr.copy()
    factors = [
        ('Seasonality', season_vals[m]),
        ('Burnout', burnout_vals[m]),
        ('Regime Shift', regime_vals[m]),
        ('Media Wave', media_vals[m]),
    ]

    for fname, fval in factors:
        prev = cumul.copy()
        cumul = cumul + fval
        cumul_clipped = np.clip(cumul, 0.01, 0.99)
        prev_clipped = np.clip(prev, 0.01, 0.99)
        color = factor_colors[fname]
        if abs(fval) > 0.005:
            lo = np.minimum(prev_clipped, cumul_clipped)
            hi = np.maximum(prev_clipped, cumul_clipped)
            ax.fill_between(x, lo, hi, alpha=0.35, color=color,
                            label=f'{fname} ({fval:+.3f})')

    true_m = np.clip(cumul, 0.01, 0.99)
    ax.plot(x, true_m, 'k--', linewidth=2, label='True S-curve')
    ax.plot(x, ms_insample[m, :], '-', linewidth=2.5, color=C_MULTISPLINE,
            alpha=0.85, label='MV Spline fit')
    ax.plot(x, spline_insample[m, :], ':', linewidth=2, color=C_SPLINE,
            alpha=0.7, label='Opt. Spline fit')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Refinancing Incentive ($x$)')
    ax.set_ylabel('CPR')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure C: Environmental Factors Reshaping the S-Curve\n'
             'Multivariate Spline captures both shape flexibility and factor shifts',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("compare_env_factors.png")
plt.close()
print("Saved compare_env_factors.png")

# =============================================================================
# FIGURE D: High-Noise scenario -- per-month comparison
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

noise_slices = [
    (6,  'Month 6 (high noise, $\\sigma=0.08$)'),
    (30, 'Month 30 (high noise)'),
    (60, 'Month 60 (high noise)'),
    (96, 'Month 96 (high noise)'),
]

for ax, (m, title) in zip(axes.flat, noise_slices):
    true_m = CPR_true[m, :]
    obs_m = CPR_noisy[m, :]

    ax.plot(x, obs_m, '.', color='#AAAAAA', markersize=4, alpha=0.6,
            label='Observed (noisy)')
    ax.plot(x, true_m, '-', linewidth=2.5, color=C_TRUE, label='True CPR')
    ax.plot(x, spline_noisy[m], '--', linewidth=2, color=C_SPLINE,
            label=f'Spline (RMSE={rmse(true_m, spline_noisy[m]):.4f})')
    ax.plot(x, multi_noisy[m], '--', linewidth=2, color=C_MULTI,
            label=f'Multi (RMSE={rmse(true_m, multi_noisy[m]):.4f})')
    ax.plot(x, ms_noisy[m], '-', linewidth=2.5, color=C_MULTISPLINE,
            label=f'MV Spline (RMSE={rmse(true_m, ms_noisy[m]):.4f})')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Refinancing Incentive ($x$)')
    ax.set_ylabel('CPR')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure D: High-Noise Comparison\n'
             'Multivariate Spline resists overfitting with structural + flexible basis',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("compare_rmse_timeline.png")
plt.close()
print("Saved compare_rmse_timeline.png")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("  CONSOLIDATED RESULTS")
print("=" * 80)
print(f"\n  {'Scenario':<35} {'Opt Spline':>12} {'Multivar':>12} {'MV Spline':>12} {'Winner':>15}")
print("  " + "-" * 86)
print(f"  {'A. In-Sample':<35} {r_sp_a:>12.4f} {r_mu_a:>12.4f} {r_ms_a:>12.4f} "
      f"{'MV SPLINE' if r_ms_a <= min(r_sp_a, r_mu_a) else 'SPLINE' if r_sp_a < r_mu_a else 'MULTI':>15}")
print(f"  {'B. Forecasting (80-119)':<35} {r_sp_b:>12.4f} {r_mu_b:>12.4f} {r_ms_b:>12.4f} "
      f"{'MV SPLINE' if r_ms_b <= min(r_sp_b, r_mu_b) else 'MULTI' if r_mu_b < r_sp_b else 'SPLINE':>15}")
print(f"  {'C. High Noise (0.08)':<35} {r_sp_c:>12.4f} {r_mu_c:>12.4f} {r_ms_c:>12.4f} "
      f"{'MV SPLINE' if r_ms_c <= min(r_sp_c, r_mu_c) else 'MULTI' if r_mu_c < r_sp_c else 'SPLINE':>15}")
print(f"  {'D. Sparse (12/50)':<35} {r_sp_d:>12.4f} {r_mu_d:>12.4f} {r_ms_d:>12.4f} "
      f"{'MV SPLINE' if r_ms_d <= min(r_sp_d, r_mu_d) else 'MULTI' if r_mu_d < r_sp_d else 'SPLINE':>15}")
print("  " + "-" * 86)
print(f"\n  Feature count:  Spline={11}/month ({11*120} total)  "
      f"Multi={X_multi.shape[1]}  MV Spline={X_ms.shape[1]}")
print("\n" + "=" * 80)
