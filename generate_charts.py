import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, make_lsq_spline
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

np.random.seed(42)

# =============================================================================
# 1. SIMULATION PARAMETERS
# =============================================================================
n_months = 120  # 10 years of monthly data
n_incentive = 50  # refinancing incentive grid points

# Refinancing incentive grid
x = np.linspace(0, 1, n_incentive)

# Time grid (months)
t_months = np.arange(n_months)
t_years = t_months / 12.0

# Logistic S-curve parameters
L = 1.0
k_steep = 6.0
x0 = 0.5

# =============================================================================
# 2. EXTERNAL FACTOR DEFINITIONS
# =============================================================================

# --- Seasonality ---
# Prepayments peak in summer (month ~6-8) due to home buying season
def seasonality_factor(month_indices, amplitude=0.08):
    """Monthly seasonality with summer peak."""
    return amplitude * np.sin(2 * np.pi * (month_indices - 3) / 12)

# --- Burnout ---
# Borrowers who could refinance but haven't become less likely over time
def burnout_factor(age_months, rate=0.03, max_burnout=0.25):
    """Exponential burnout: reduces CPR as pool ages."""
    return -max_burnout * (1 - np.exp(-rate * age_months))

# --- Regime Shift ---
# Structural break at month 60 (e.g., post-crisis tightening)
def regime_factor(month_indices, shift_month=60, shift_magnitude=-0.10, transition_width=3):
    """Sigmoid regime shift at a given month."""
    return shift_magnitude / (1 + np.exp(-(month_indices - shift_month) / transition_width))

# --- Media / Refi Wave ---
# Temporary spike in awareness drives prepayments
def media_factor(month_indices, peak_month=30, amplitude=0.06, width=4):
    """Gaussian-shaped media/refi wave effect."""
    return amplitude * np.exp(-0.5 * ((month_indices - peak_month) / width) ** 2)

# =============================================================================
# 3. GENERATE FULL PANEL DATA (incentive x time)
# =============================================================================

# Base S-curve (static)
base_cpr = L / (1 + np.exp(-k_steep * (x - x0)))
base_cpr = 0.05 + 0.9 * base_cpr
base_cpr = np.clip(base_cpr, 0, 1)

# Create panel: CPR[month, incentive]
CPR_true = np.zeros((n_months, n_incentive))
season_vals = seasonality_factor(t_months)
burnout_vals = burnout_factor(t_months)
regime_vals = regime_factor(t_months)
media_vals = media_factor(t_months)

for m in range(n_months):
    adjustment = season_vals[m] + burnout_vals[m] + regime_vals[m] + media_vals[m]
    CPR_true[m, :] = base_cpr + adjustment
CPR_true = np.clip(CPR_true, 0.01, 0.99)

# Noisy observations
noise_sigma = 0.025
CPR_obs = CPR_true + np.random.normal(0, noise_sigma, size=CPR_true.shape)
CPR_obs = np.clip(CPR_obs, 0.01, 0.99)

# =============================================================================
# 4. CROSS-SECTIONAL MODELS (single time slice, for backward compat)
# =============================================================================

# Use month 24 as representative cross-section
t_cross = 24
true_cpr = CPR_true[t_cross, :]
obs_cpr = CPR_obs[t_cross, :]

# Bayesian baseline (exponential smoothing)
alpha = 0.2
bayes_cpr = np.zeros_like(obs_cpr)
bayes_cpr[0] = obs_cpr[0]
for i in range(1, len(obs_cpr)):
    bayes_cpr[i] = alpha * obs_cpr[i] + (1 - alpha) * bayes_cpr[i - 1]
bayes_cpr = np.maximum.accumulate(bayes_cpr)

# Monotone PCHIP
pchip_cpr = PchipInterpolator(x, obs_cpr)(x)

# Optimized monotone spline
k_spline = 3
knots = np.linspace(x[0], x[-1], 8)
knots = np.concatenate(([x[0]] * k_spline, knots, [x[-1]] * k_spline))
spline_cpr = make_lsq_spline(x, obs_cpr, knots, k=k_spline)(x)
spline_cpr = np.maximum.accumulate(spline_cpr)

# RMSE cross-sectional
rmse_bayes = np.sqrt(mean_squared_error(true_cpr, bayes_cpr))
rmse_pchip = np.sqrt(mean_squared_error(true_cpr, pchip_cpr))
rmse_spline = np.sqrt(mean_squared_error(true_cpr, spline_cpr))

print("=== Cross-Sectional RMSE (Month 24) ===")
print(f"  Bayesian baseline: {rmse_bayes:.4f}")
print(f"  PCHIP:             {rmse_pchip:.4f}")
print(f"  Optimized Spline:  {rmse_spline:.4f}")

# =============================================================================
# 5. MULTIVARIATE REGRESSION WITH EXTERNAL FACTORS
# =============================================================================

def build_design_matrix(x_grid, t_grid):
    """
    Build design matrix for multivariate CPR regression.
    
    Features per (x, t) observation:
      1. Logistic S-curve basis: sigma(k*(x - x0)) for several k values
      2. Seasonality: sin(2*pi*t/12), cos(2*pi*t/12)
      3. Burnout: 1 - exp(-rate * t)
      4. Regime shift: sigmoid at shift_month
      5. Interactions: S-curve * burnout, S-curve * regime
    """
    n_t = len(t_grid)
    n_x = len(x_grid)
    N = n_t * n_x
    
    # Flatten panel
    X_flat = np.tile(x_grid, n_t)  # incentive for each obs
    T_flat = np.repeat(t_grid, n_x)  # time for each obs
    
    features = []
    feature_names = []
    
    # 1. Intercept
    features.append(np.ones(N))
    feature_names.append('intercept')
    
    # 2. S-curve basis (logistic at different steepnesses)
    for kk in [4, 6, 10]:
        feat = 1.0 / (1 + np.exp(-kk * (X_flat - 0.5)))
        features.append(feat)
        feature_names.append(f'logistic_k{kk}')
    
    # 3. Quadratic incentive term
    features.append(X_flat ** 2)
    feature_names.append('x_squared')
    
    # 4. Seasonality (Fourier terms)
    features.append(np.sin(2 * np.pi * T_flat / 12))
    feature_names.append('sin_season')
    features.append(np.cos(2 * np.pi * T_flat / 12))
    feature_names.append('cos_season')
    
    # 5. Second harmonic seasonality
    features.append(np.sin(4 * np.pi * T_flat / 12))
    feature_names.append('sin_season_2')
    features.append(np.cos(4 * np.pi * T_flat / 12))
    feature_names.append('cos_season_2')
    
    # 6. Burnout
    burnout = 1 - np.exp(-0.03 * T_flat)
    features.append(burnout)
    feature_names.append('burnout')
    
    # 7. Regime shift (sigmoid)
    regime = 1.0 / (1 + np.exp(-(T_flat - 60) / 3))
    features.append(regime)
    feature_names.append('regime')
    
    # 8. Interaction: burnout * S-curve
    s_curve_main = 1.0 / (1 + np.exp(-6 * (X_flat - 0.5)))
    features.append(burnout * s_curve_main)
    feature_names.append('burnout_x_scurve')
    
    # 9. Interaction: regime * S-curve
    features.append(regime * s_curve_main)
    feature_names.append('regime_x_scurve')
    
    # 10. Linear time trend
    features.append(T_flat / n_months)
    feature_names.append('time_trend')
    
    X_mat = np.column_stack(features)
    return X_mat, feature_names, X_flat, T_flat

# Build design matrix
X_mat, feat_names, X_flat, T_flat = build_design_matrix(x, t_months)
y_flat = CPR_obs.flatten()
y_true_flat = CPR_true.flatten()

# OLS regression with ridge regularization
lambda_ridge = 0.01
XtX = X_mat.T @ X_mat + lambda_ridge * np.eye(X_mat.shape[1])
Xty = X_mat.T @ y_flat
beta_hat = np.linalg.solve(XtX, Xty)

# Predicted CPR
y_pred_multi = X_mat @ beta_hat
y_pred_multi = np.clip(y_pred_multi, 0.01, 0.99)
CPR_pred_multi = y_pred_multi.reshape(n_months, n_incentive)

# Enforce monotonicity in x for each time slice
for m in range(n_months):
    CPR_pred_multi[m, :] = np.maximum.accumulate(CPR_pred_multi[m, :])

rmse_multi = np.sqrt(mean_squared_error(y_true_flat, y_pred_multi.flatten()))

# =============================================================================
# 5b. PANEL-WIDE EVALUATION OF SINGLE-VARIABLE METHODS (fair comparison)
# =============================================================================
# Apply each single-variable method independently at every month
CPR_bayes_panel = np.zeros_like(CPR_true)
CPR_pchip_panel = np.zeros_like(CPR_true)
CPR_spline_panel = np.zeros_like(CPR_true)

for m in range(n_months):
    obs_m = CPR_obs[m, :]

    # Bayesian baseline
    b = np.zeros_like(obs_m)
    b[0] = obs_m[0]
    for i in range(1, len(obs_m)):
        b[i] = alpha * obs_m[i] + (1 - alpha) * b[i - 1]
    CPR_bayes_panel[m, :] = np.maximum.accumulate(b)

    # PCHIP
    CPR_pchip_panel[m, :] = PchipInterpolator(x, obs_m)(x)

    # Spline
    s = make_lsq_spline(x, obs_m, knots, k=k_spline)(x)
    CPR_spline_panel[m, :] = np.maximum.accumulate(s)

rmse_bayes_panel = np.sqrt(mean_squared_error(CPR_true.flatten(), CPR_bayes_panel.flatten()))
rmse_pchip_panel = np.sqrt(mean_squared_error(CPR_true.flatten(), CPR_pchip_panel.flatten()))
rmse_spline_panel = np.sqrt(mean_squared_error(CPR_true.flatten(), CPR_spline_panel.flatten()))

# Also compute multivariate cross-sectional RMSE at month 24 for fair table
rmse_multi_cross = np.sqrt(mean_squared_error(CPR_true[t_cross, :], CPR_pred_multi[t_cross, :]))

print(f"\n=== Panel-Wide RMSE (all 120 months, FAIR comparison) ===")
print(f"  Bayesian baseline:      {rmse_bayes_panel:.4f}")
print(f"  PCHIP:                  {rmse_pchip_panel:.4f}")
print(f"  Optimized Spline:       {rmse_spline_panel:.4f}")
print(f"  Multivariate Regression:{rmse_multi:.4f}")

print(f"\n=== Cross-Sectional RMSE at Month 24 (all methods) ===")
print(f"  Bayesian baseline:      {rmse_bayes:.4f}")
print(f"  PCHIP:                  {rmse_pchip:.4f}")
print(f"  Optimized Spline:       {rmse_spline:.4f}")
print(f"  Multivariate Regression:{rmse_multi_cross:.4f}")

print(f"\n=== Multivariate Regression Coefficients ===")
for name, coef in zip(feat_names, beta_hat):
    print(f"    {name:25s}: {coef:+.6f}")

# =============================================================================
# 6. ABLATION STUDY - contribution of each factor
# =============================================================================

def fit_subset(feature_indices, X_mat, y_flat, y_true_flat, lambda_r=0.01):
    """Fit model using only specified feature columns."""
    X_sub = X_mat[:, feature_indices]
    XtX = X_sub.T @ X_sub + lambda_r * np.eye(X_sub.shape[1])
    Xty = X_sub.T @ y_flat
    b = np.linalg.solve(XtX, Xty)
    pred = np.clip(X_sub @ b, 0.01, 0.99)
    return np.sqrt(mean_squared_error(y_true_flat, pred))

# Define feature groups
base_idx = [0, 1, 2, 3, 4]  # intercept + S-curve basis + x^2
season_idx = [5, 6, 7, 8]   # Fourier terms
burnout_idx = [9]            # burnout
regime_idx = [10]            # regime
interaction_idx = [11, 12]   # interactions
trend_idx = [13]             # time trend

ablation_results = {}
ablation_results['Base S-curve only'] = fit_subset(base_idx, X_mat, y_flat, y_true_flat)
ablation_results['+ Seasonality'] = fit_subset(base_idx + season_idx, X_mat, y_flat, y_true_flat)
ablation_results['+ Burnout'] = fit_subset(base_idx + burnout_idx, X_mat, y_flat, y_true_flat)
ablation_results['+ Regime'] = fit_subset(base_idx + regime_idx, X_mat, y_flat, y_true_flat)
ablation_results['+ Season + Burnout'] = fit_subset(base_idx + season_idx + burnout_idx, X_mat, y_flat, y_true_flat)
ablation_results['Full model'] = rmse_multi

print(f"\n=== Ablation Study ===")
for name, rmse in ablation_results.items():
    print(f"  {name:30s}: RMSE = {rmse:.4f}")

# =============================================================================
# FIGURE 1: CPR Fit Comparison (cross-sectional)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, true_cpr, 'k-', linewidth=2.5, label='True CPR (logistic)')
ax.plot(x, obs_cpr, 'o', color='#888888', markersize=4, alpha=0.6, label='Observed CPR (noisy)')
ax.plot(x, bayes_cpr, '-', linewidth=2, color='#E74C3C',
        label=f'Bayesian Baseline (RMSE={rmse_bayes:.4f})')
ax.plot(x, pchip_cpr, '--', linewidth=2, color='#3498DB',
        label=f'Monotone PCHIP (RMSE={rmse_pchip:.4f})')
ax.plot(x, spline_cpr, '-.', linewidth=2, color='#2ECC71',
        label=f'Optimized Spline (RMSE={rmse_spline:.4f})')
ax.plot(x, CPR_pred_multi[t_cross, :], '-', linewidth=2.5, color='#9B59B6',
        label=f'Multivariate Regression (RMSE={np.sqrt(mean_squared_error(true_cpr, CPR_pred_multi[t_cross,:])):.4f})')
ax.set_xlabel('Refinancing Incentive ($x$)')
ax.set_ylabel('Conditional Prepayment Rate (CPR)')
ax.set_title('Figure 1: CPR S-Curve Estimation \u2014 Method Comparison (Month 24)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig("fig1_cpr_fit.png")
plt.close()
print("\nSaved fig1_cpr_fit.png")

# =============================================================================
# FIGURE 2: Implied OAS Comparison
# =============================================================================
oas_true = 100 - 80 * true_cpr
oas_bayes = 100 - 80 * bayes_cpr
oas_pchip = 100 - 80 * pchip_cpr
oas_spline = 100 - 80 * spline_cpr
oas_multi = 100 - 80 * CPR_pred_multi[t_cross, :]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, oas_true, 'k-', linewidth=2.5, label='OAS (True CPR)')
ax.plot(x, oas_bayes, '-', linewidth=2, color='#E74C3C', label='OAS (Bayesian)')
ax.plot(x, oas_pchip, '--', linewidth=2, color='#3498DB', label='OAS (PCHIP)')
ax.plot(x, oas_spline, '-.', linewidth=2, color='#2ECC71', label='OAS (Optimized Spline)')
ax.plot(x, oas_multi, '-', linewidth=2.5, color='#9B59B6', label='OAS (Multivariate)')
ax.set_xlabel('Refinancing Incentive ($x$)')
ax.set_ylabel('Option-Adjusted Spread (bps)')
ax.set_title('Figure 2: Implied OAS from Estimated CPR Curves')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper right', framealpha=0.9)
ax.set_xlim([0, 1])
plt.tight_layout()
plt.savefig("fig2_oas_comparison.png")
plt.close()
print("Saved fig2_oas_comparison.png")

# =============================================================================
# FIGURE 3: Residual Analysis
# =============================================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)

residuals = {
    'Bayesian': (bayes_cpr - true_cpr, '#E74C3C'),
    'PCHIP': (pchip_cpr - true_cpr, '#3498DB'),
    'Spline': (spline_cpr - true_cpr, '#2ECC71'),
    'Multivariate': (CPR_pred_multi[t_cross, :] - true_cpr, '#9B59B6'),
}

for ax, (name, (resid, color)) in zip(axes, residuals.items()):
    ax.bar(x, resid, width=0.015, color=color, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Refinancing Incentive ($x$)')
    ax.set_title(name)
    ax.grid(True, linestyle='--', alpha=0.3)

axes[0].set_ylabel('Residual (Est. \u2212 True)')
fig.suptitle('Figure 3: Residual Analysis by Method', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("fig3_residuals.png")
plt.close()
print("Saved fig3_residuals.png")

# =============================================================================
# FIGURE 4: External Factor Decomposition
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Seasonality
ax = axes[0, 0]
ax.plot(t_months, season_vals, '-', color='#E67E22', linewidth=2)
ax.set_xlabel('Month')
ax.set_ylabel('CPR Adjustment')
ax.set_title('(a) Seasonality Effect')
ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
ax.grid(True, linestyle='--', alpha=0.3)

# Burnout
ax = axes[0, 1]
ax.plot(t_months, burnout_vals, '-', color='#8E44AD', linewidth=2)
ax.set_xlabel('Month')
ax.set_ylabel('CPR Adjustment')
ax.set_title('(b) Burnout Effect')
ax.grid(True, linestyle='--', alpha=0.3)

# Regime shift
ax = axes[1, 0]
ax.plot(t_months, regime_vals, '-', color='#C0392B', linewidth=2)
ax.axvline(60, color='gray', linewidth=1, linestyle=':', label='Regime break')
ax.set_xlabel('Month')
ax.set_ylabel('CPR Adjustment')
ax.set_title('(c) Regime Shift Effect')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)

# Media / refi wave
ax = axes[1, 1]
ax.plot(t_months, media_vals, '-', color='#27AE60', linewidth=2)
ax.set_xlabel('Month')
ax.set_ylabel('CPR Adjustment')
ax.set_title('(d) Media / Refi Wave Effect')
ax.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure 4: External Factor Decomposition', fontsize=14)
plt.tight_layout()
plt.savefig("fig4_factor_decomposition.png")
plt.close()
print("Saved fig4_factor_decomposition.png")

# =============================================================================
# FIGURE 5: Time-Varying S-Curves
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))
cmap = plt.cm.viridis
month_samples = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108]

for i, m in enumerate(month_samples):
    color = cmap(i / len(month_samples))
    ax.plot(x, CPR_true[m, :], '-', color=color, linewidth=1.5, alpha=0.7,
            label=f'Month {m}')
    ax.plot(x, CPR_pred_multi[m, :], '--', color=color, linewidth=1.5, alpha=0.9)

# Dummy entries for legend
ax.plot([], [], 'k-', linewidth=1.5, label='True (solid)')
ax.plot([], [], 'k--', linewidth=1.5, label='Predicted (dashed)')

ax.set_xlabel('Refinancing Incentive ($x$)')
ax.set_ylabel('CPR')
ax.set_title('Figure 5: Time-Varying S-Curves \u2014 True vs. Multivariate Predicted')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig("fig5_time_varying_scurves.png")
plt.close()
print("Saved fig5_time_varying_scurves.png")

# =============================================================================
# FIGURE 6: Derivative / Marginal CPR
# =============================================================================
dx = x[1] - x[0]
deriv_true = np.gradient(true_cpr, dx)
deriv_bayes = np.gradient(bayes_cpr, dx)
deriv_pchip = np.gradient(pchip_cpr, dx)
deriv_spline = np.gradient(spline_cpr, dx)
deriv_multi = np.gradient(CPR_pred_multi[t_cross, :], dx)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, deriv_true, 'k-', linewidth=2.5, label='True dCPR/dx')
ax.plot(x, deriv_bayes, '-', linewidth=2, color='#E74C3C', label='Bayesian dCPR/dx')
ax.plot(x, deriv_pchip, '--', linewidth=2, color='#3498DB', label='PCHIP dCPR/dx')
ax.plot(x, deriv_spline, '-.', linewidth=2, color='#2ECC71', label='Spline dCPR/dx')
ax.plot(x, deriv_multi, '-', linewidth=2, color='#9B59B6', label='Multivariate dCPR/dx')
ax.set_xlabel('Refinancing Incentive ($x$)')
ax.set_ylabel('$dCPR/dx$')
ax.set_title('Figure 6: Marginal CPR (First Derivative)')
ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='best', framealpha=0.9)
plt.tight_layout()
plt.savefig("fig6_derivative.png")
plt.close()
print("Saved fig6_derivative.png")

# =============================================================================
# FIGURE 7: Ablation Study Bar Chart
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
names = list(ablation_results.keys())
rmses = list(ablation_results.values())
colors = ['#95A5A6', '#E67E22', '#8E44AD', '#C0392B', '#3498DB', '#2ECC71']
bars = ax.barh(names, rmses, color=colors, edgecolor='white', height=0.6)
ax.set_xlabel('RMSE')
ax.set_title('Figure 7: Ablation Study \u2014 Contribution of Each Factor')
ax.invert_yaxis()
for bar, rmse in zip(bars, rmses):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{rmse:.4f}', va='center', fontsize=10)
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("fig7_ablation.png")
plt.close()
print("Saved fig7_ablation.png")

# =============================================================================
# FIGURE 8: Seasonality Impact on CPR Time Series
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 5))

# Pick mid-incentive (x=0.5) to show time series
mid_idx = n_incentive // 2
ax.plot(t_months, CPR_true[:, mid_idx], 'k-', linewidth=2, label='True CPR')
ax.plot(t_months, CPR_obs[:, mid_idx], 'o', color='#888', markersize=3, alpha=0.5,
        label='Observed CPR')
ax.plot(t_months, CPR_pred_multi[:, mid_idx], '-', linewidth=2, color='#9B59B6',
        label='Multivariate Predicted')

# Mark regime shift
ax.axvline(60, color='red', linewidth=1.5, linestyle=':', alpha=0.7, label='Regime shift')
ax.axvline(30, color='green', linewidth=1.5, linestyle=':', alpha=0.7, label='Media wave peak')

ax.set_xlabel('Month')
ax.set_ylabel('CPR')
ax.set_title(f'Figure 8: CPR Time Series at Mid-Incentive ($x = {x[mid_idx]:.2f}$)')
ax.legend(loc='best', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("fig8_timeseries.png")
plt.close()
print("Saved fig8_timeseries.png")

# =============================================================================
# FIGURE 9: Noise Sensitivity
# =============================================================================
noise_levels = [0.01, 0.02, 0.05, 0.10]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, sigma in zip(axes.flat, noise_levels):
    np.random.seed(42)
    obs_noisy = true_cpr + np.random.normal(0, sigma, size=true_cpr.shape)
    obs_noisy = np.clip(obs_noisy, 0, 1)

    b = np.zeros_like(obs_noisy)
    b[0] = obs_noisy[0]
    for i in range(1, len(obs_noisy)):
        b[i] = alpha * obs_noisy[i] + (1 - alpha) * b[i - 1]
    b = np.maximum.accumulate(b)

    p = PchipInterpolator(x, obs_noisy)(x)

    s = make_lsq_spline(x, obs_noisy, knots, k=k_spline)(x)
    s = np.maximum.accumulate(s)

    ax.plot(x, true_cpr, 'k-', linewidth=2, label='True')
    ax.plot(x, obs_noisy, 'o', color='#888', markersize=3, alpha=0.5, label='Observed')
    ax.plot(x, b, '-', color='#E74C3C', linewidth=1.5,
            label=f'Bayesian (RMSE={np.sqrt(mean_squared_error(true_cpr, b)):.4f})')
    ax.plot(x, p, '--', color='#3498DB', linewidth=1.5,
            label=f'PCHIP (RMSE={np.sqrt(mean_squared_error(true_cpr, p)):.4f})')
    ax.plot(x, s, '-.', color='#2ECC71', linewidth=1.5,
            label=f'Spline (RMSE={np.sqrt(mean_squared_error(true_cpr, s)):.4f})')
    ax.set_title(f'$\\sigma = {sigma}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('CPR')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure 9: Sensitivity to Observation Noise Level ($\\sigma$)', fontsize=14)
plt.tight_layout()
plt.savefig("fig9_noise_sensitivity.png")
plt.close()
print("Saved fig9_noise_sensitivity.png")

# =============================================================================
# FIGURE 10: Steepness Sensitivity
# =============================================================================
steepness_vals = [3, 6, 10, 15]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, kval in zip(axes.flat, steepness_vals):
    true_k = L / (1 + np.exp(-kval * (x - x0)))
    true_k = 0.05 + 0.9 * true_k
    true_k = np.clip(true_k, 0, 1)

    np.random.seed(42)
    obs_k = true_k + np.random.normal(0, 0.02, size=true_k.shape)
    obs_k = np.clip(obs_k, 0, 1)

    b = np.zeros_like(obs_k)
    b[0] = obs_k[0]
    for i in range(1, len(obs_k)):
        b[i] = alpha * obs_k[i] + (1 - alpha) * b[i - 1]
    b = np.maximum.accumulate(b)

    p = PchipInterpolator(x, obs_k)(x)

    s = make_lsq_spline(x, obs_k, knots, k=k_spline)(x)
    s = np.maximum.accumulate(s)

    ax.plot(x, true_k, 'k-', linewidth=2, label='True')
    ax.plot(x, obs_k, 'o', color='#888', markersize=3, alpha=0.5, label='Observed')
    ax.plot(x, b, '-', color='#E74C3C', linewidth=1.5,
            label=f'Bayesian (RMSE={np.sqrt(mean_squared_error(true_k, b)):.4f})')
    ax.plot(x, p, '--', color='#3498DB', linewidth=1.5,
            label=f'PCHIP (RMSE={np.sqrt(mean_squared_error(true_k, p)):.4f})')
    ax.plot(x, s, '-.', color='#2ECC71', linewidth=1.5,
            label=f'Spline (RMSE={np.sqrt(mean_squared_error(true_k, s)):.4f})')
    ax.set_title(f'Steepness $k = {kval}$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('CPR')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)

fig.suptitle('Figure 10: Sensitivity to S-Curve Steepness ($k$)', fontsize=14)
plt.tight_layout()
plt.savefig("fig10_steepness_sensitivity.png")
plt.close()
print("Saved fig10_steepness_sensitivity.png")

# =============================================================================
# FIGURE 11: Coefficient Importance (Forest Plot)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
# Normalize coefficients by feature std for comparable scale
feature_stds = X_mat.std(axis=0)
feature_stds[feature_stds == 0] = 1
normalized_coefs = beta_hat * feature_stds

sorted_idx = np.argsort(np.abs(normalized_coefs))
sorted_names = [feat_names[i] for i in sorted_idx]
sorted_coefs = normalized_coefs[sorted_idx]
colors_bar = ['#E74C3C' if c < 0 else '#2ECC71' for c in sorted_coefs]

ax.barh(sorted_names, sorted_coefs, color=colors_bar, edgecolor='white', height=0.6)
ax.set_xlabel('Standardized Coefficient')
ax.set_title('Figure 11: Feature Importance (Standardized Coefficients)')
ax.axvline(0, color='k', linewidth=0.8)
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig("fig11_coefficients.png")
plt.close()
print("Saved fig11_coefficients.png")

print("\n=== All figures generated successfully ===")
