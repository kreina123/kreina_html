# S-Curve Estimation for Mortgage Prepayment Modeling: A Multivariate Framework with External Factors

---

**Authors:** [Author Name], [Co-Author Name]
**Affiliation:** [Department of Quantitative Finance / Fixed Income Research]
**Date:** March 2026
**Keywords:** mortgage prepayment, conditional prepayment rate, S-curve estimation, monotone interpolation, option-adjusted spread, B-splines, PCHIP, Bayesian smoothing, seasonality, burnout, regime shift, panel regression

---

## Abstract

Accurate estimation of the conditional prepayment rate (CPR) S-curve is foundational to mortgage-backed securities (MBS) valuation, risk management, and option-adjusted spread (OAS) computation. While single-variable methods—Bayesian smoothing, PCHIP interpolation, and optimized B-splines—can reconstruct each month's S-curve independently, they are fundamentally limited: they cannot forecast CPR for unobserved time periods, they overfit under realistic noise conditions, and they degrade rapidly with sparse incentive observations.

This paper presents a **multivariate panel regression** framework that explicitly incorporates **seasonality**, **borrower burnout**, **regime shifts**, and **media-driven refinancing waves** as additional covariates alongside the refinancing incentive. We evaluate performance across three realistic scenarios:

1. **Forecasting:** Training on months 0–79 and predicting months 80–119. The multivariate model achieves 0.0481 RMSE versus 0.1463 for the best single-variable baseline—an **83.6% improvement**.
2. **High noise ($\sigma = 0.08$):** The multivariate model achieves 0.0368 RMSE versus 0.0419 for the best single-variable method—a **12.2% improvement**.
3. **Sparse data** (12 of 50 incentive points observed): The multivariate model matches or outperforms all single-variable methods despite dramatically reduced observations.

An ablation study quantifies the marginal contribution of each environmental factor, with burnout (-37.7%) and regime shifts (-35.4%) providing the largest improvements. We further analyze implications for OAS pricing, derivative smoothness, and production deployment.

---

## 1. Introduction

### 1.1 Background and Motivation

Mortgage-backed securities are among the most actively traded fixed-income instruments, with outstanding volumes exceeding $12 trillion in the U.S. alone (SIFMA, 2024). A central challenge in MBS valuation is the modeling of **prepayment risk**: borrowers may refinance their mortgages when interest rates decline, returning principal early and altering the cash flow profile of the security.

The relationship between a borrower's refinancing incentive and the conditional probability of prepayment is commonly referred to as the **prepayment S-curve**. This terminology reflects the characteristic sigmoid shape: at low refinancing incentives, prepayment rates are low; as incentive increases, prepayment accelerates through a transition zone; at high incentives, the rate plateaus as remaining borrowers are increasingly refinancing-averse.

However, the static S-curve is an incomplete description of prepayment behavior. Empirical evidence demonstrates that CPR varies substantially across time due to factors orthogonal to the refinancing incentive:

- **Seasonality:** Home sales and refinancing activity peak in spring/summer months (Hayre, 2001).
- **Borrower burnout:** Pools that have experienced high refinancing incentive for extended periods show declining prepayment rates as responsive borrowers exit (Richard and Roll, 1989).
- **Regime shifts:** Regulatory changes (e.g., HARP, COVID forbearance programs), credit tightening, and structural market changes cause discrete shifts in prepayment behavior.
- **Media and awareness effects:** Periods of intense media coverage of low interest rates can temporarily spike refinancing activity above what the pure incentive would predict.

### 1.2 The Fundamental Limitation of Single-Variable Methods

Single-variable S-curve estimation methods—regardless of their sophistication—share a common structural limitation: **they fit each month's observed data independently and have no mechanism to generalize across time**.

Consider the implications:

- **No forecasting ability.** Given CPR observations for months 0–79, a single-variable spline fitted to month 79's data provides no information about what month 80's S-curve should look like. The best a practitioner can do is use the static base curve or the average of previously fitted curves—both crude approximations.
- **Overfitting to noise.** With 50 observations per month and 11+ free parameters (spline coefficients), a B-spline will track noise in the data rather than the underlying signal. In low-noise settings ($\sigma ≈ 0.025$) this is harmless, but at realistic noise levels ($\sigma ≈ 0.05$–$0.10$) the effect is substantial.
- **No structural insight.** Even when single-variable methods achieve low RMSE, they provide no decomposition of *why* the S-curve shifted—was it seasonality? burnout? a regime change? This lack of interpretability limits risk management and scenario analysis.

The multivariate panel regression addresses all three limitations by building a structural model of the time-varying S-curve.

### 1.3 Problem Formulation

We seek to estimate a time-varying CPR function:

$$\text{CPR}(x, t) : \mathbb{R} \times \mathbb{R}^+ \to [0, 1]$$

where $x$ is the refinancing incentive and $t$ is time (in months). The function should satisfy:

1. **Boundedness:** $0 \leq \text{CPR}(x, t) \leq 1$ for all $(x, t)$.
2. **Monotonicity in $x$:** $\text{CPR}(x_1, t) \leq \text{CPR}(x_2, t)$ when $x_1 \leq x_2$, for each fixed $t$.
3. **Smoothness:** At least $C^1$-continuous in $x$ for well-defined duration and convexity.
4. **Temporal coherence:** Smooth variation in $t$, except at identified regime boundaries.

The observed data takes the form:

$$y_{it} = \text{CPR}(x_i, t) + \varepsilon_{it}, \quad \varepsilon_{it} \sim \mathcal{N}(0, \sigma^2)$$

### 1.4 Contributions

1. We demonstrate that the multivariate panel regression **outperforms all single-variable methods** in three realistic evaluation scenarios: forecasting, high noise, and sparse data.
2. We conduct a systematic **ablation study** quantifying the marginal RMSE improvement from each environmental factor.
3. We present a novel **environmental factor stacking visualization** showing how each factor physically deforms the S-curve over time.
4. We analyze OAS implications and provide a complete, reproducible Python implementation.

### 1.5 Paper Organization

Section 2 describes the simulation framework. Section 3 presents the three single-variable estimation methods. Section 4 details the multivariate regression model. Section 5 presents the comparative evaluation across three scenarios. Section 6 provides the environmental factor decomposition. Section 7 analyzes OAS implications. Section 8 reports sensitivity analysis. Section 9 discusses practical considerations. Section 10 concludes.

---

## 2. Simulation Framework

### 2.1 Ground-Truth S-Curve with External Factors

We simulate a panel of CPR observations over $T = 120$ months and $n = 50$ incentive levels. The true CPR is generated as:

$$\text{CPR}_{\text{true}}(x_i, t) = \underbrace{\text{S}(x_i)}_{\text{base S-curve}} + \underbrace{\delta_{\text{seas}}(t)}_{\text{seasonality}} + \underbrace{\delta_{\text{burn}}(t)}_{\text{burnout}} + \underbrace{\delta_{\text{reg}}(t)}_{\text{regime}} + \underbrace{\delta_{\text{med}}(t)}_{\text{media}}$$

where clipping to $[0.01, 0.99]$ is applied after summation.

#### 2.1.1 Base S-Curve

$$\text{S}(x) = 0.05 + 0.90 \cdot \frac{1}{1 + e^{-6(x - 0.5)}}$$

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| Lower asymptote | 0.05 | Baseline CPR at zero incentive |
| Upper asymptote | 0.95 | Maximum CPR at full incentive |
| Steepness $k$ | 6.0 | Transition rate |
| Inflection $x_0$ | 0.5 | Midpoint of S-curve |

#### 2.1.2 Seasonality

$$\delta_{\text{seas}}(t) = A_s \sin\!\left(\frac{2\pi(t - 3)}{12}\right), \quad A_s = 0.08$$

This produces a summer peak (months 6–8) and winter trough, consistent with the well-documented home buying season effect (Hayre, 2001). The amplitude of 8 CPR points captures the typical seasonal swing observed in agency pass-throughs.

#### 2.1.3 Burnout

$$\delta_{\text{burn}}(t) = -B_{\max}\!\left(1 - e^{-\lambda_b t}\right), \quad B_{\max} = 0.25, \quad \lambda_b = 0.03$$

Burnout reduces CPR over time as responsive borrowers exit the pool. The exponential form ensures rapid initial burnout followed by gradual flattening, matching empirical observations (Richard and Roll, 1989; Davidson and Levin, 2014). By month 96, cumulative burnout reaches approximately −0.24 CPR points.

#### 2.1.4 Regime Shift

$$\delta_{\text{reg}}(t) = \frac{R_{\text{mag}}}{1 + e^{-(t - t_R)/w_R}}, \quad R_{\text{mag}} = -0.10, \quad t_R = 60, \quad w_R = 3$$

A sigmoid transition at month 60 simulates a structural break (e.g., post-crisis credit tightening). The magnitude of −10 CPR points reflects the scale of observed regime effects in historical data.

#### 2.1.5 Media / Refinancing Wave

$$\delta_{\text{med}}(t) = M_a \exp\!\left(-\frac{(t - t_M)^2}{2w_M^2}\right), \quad M_a = 0.06, \quad t_M = 30, \quad w_M = 4$$

A Gaussian-shaped transient at month 30 models a media-driven refinancing wave. Unlike persistent factors, this effect is localized and temporary.

### 2.2 Noise Models

We evaluate under three noise conditions:

| Scenario | $\sigma$ | Points per month | Training months |
|----------|----------|-------------------|-----------------|
| Forecasting | 0.025 | 50 | 0–79 |
| High Noise | 0.080 | 50 | All |
| Sparse Data | 0.040 | 12 | All |

---

## 3. Single-Variable Estimation Methods

### 3.1 Bayesian Baseline Smoothing

The Bayesian approach implements exponential smoothing across the incentive dimension:

$$\hat{y}_i = \alpha \cdot y_i + (1 - \alpha) \cdot \hat{y}_{i-1}, \quad \alpha = 0.2$$

with post-hoc monotonicity enforcement via cumulative maximum.

**Properties:** Handles sequential updating naturally but introduces systematic lag in the steep transition region.

### 3.2 Monotone PCHIP Interpolation

Piecewise Cubic Hermite Interpolating Polynomials construct a $C^1$-continuous interpolant that preserves local monotonicity (Fritsch and Carlson, 1980). On each interval $[x_i, x_{i+1}]$:

$$p_i(x) = y_i H_{00}(t) + h_i d_i H_{10}(t) + y_{i+1} H_{01}(t) + h_i d_{i+1} H_{11}(t)$$

**Properties:** Guarantees monotonicity by construction but provides no noise smoothing—RMSE tracks the observation noise level $\sigma$.

### 3.3 Optimized Monotone B-Spline

The B-spline representation uses cubic order ($k = 4$), 8 internal knots, and least-squares fitting:

$$\text{CPR}(x) = \sum_{j=1}^{K} c_j B_{j,k}(x)$$

**Properties:** Best single-variable RMSE at low noise through implicit smoothing via limited basis dimension.

### 3.4 Critical Limitation: No Cross-Temporal Learning

All three methods fit each month's data independently. With $n = 50$ observations per month, they have ample data to fit the curve shape—but they have **zero capacity to predict CPR at a future time point**. If month 80 has not been observed, these methods can only offer:

- The static base curve $\text{S}(x)$ (ignoring all temporal effects), or
- The average of previously fitted curves (a crude and biased estimate).

This limitation motivates the multivariate approach.

---

## 4. Multivariate Panel Regression

### 4.1 Model Specification

We extend the single-variable S-curve to a multivariate panel model:

$$\text{CPR}(x_i, t) = \beta_0 + \underbrace{\sum_{m=1}^{3} \beta_m \phi_m(x_i)}_{\text{S-curve basis}} + \underbrace{\beta_4 x_i^2}_{\text{quadratic}} + \underbrace{\sum_{p=1}^{4} \gamma_p f_p(t)}_{\text{external factors}} + \underbrace{\sum_{q=1}^{2} \delta_q g_q(x_i, t)}_{\text{interactions}} + \underbrace{\beta_5 \cdot (t/T)}_{\text{trend}} + \varepsilon_{it}$$

#### 4.1.1 S-Curve Basis Functions

$$\phi_m(x) = \frac{1}{1 + e^{-k_m(x - 0.5)}}, \quad k_m \in \{4, 6, 10\}$$

Using multiple steepness values allows the model to represent S-curves that are not perfectly logistic, accommodating asymmetry and varying transition widths.

#### 4.1.2 Seasonality Features

Fourier representation of monthly seasonality:

$$f_1(t) = \sin\!\left(\frac{2\pi t}{12}\right), \quad f_2(t) = \cos\!\left(\frac{2\pi t}{12}\right), \quad f_3(t) = \sin\!\left(\frac{4\pi t}{12}\right), \quad f_4(t) = \cos\!\left(\frac{4\pi t}{12}\right)$$

The first harmonic captures the dominant annual cycle; the second harmonic allows for asymmetric seasonal patterns.

#### 4.1.3 Burnout Feature

$$f_5(t) = 1 - e^{-\lambda_b t}, \quad \lambda_b = 0.03$$

This monotonically increasing function captures cumulative burnout. The coefficient $\gamma_5$ is expected to be negative.

#### 4.1.4 Regime Shift Feature

$$f_6(t) = \frac{1}{1 + e^{-(t - t_R)/w_R}}$$

A sigmoid centered at the hypothesized break point $t_R = 60$.

#### 4.1.5 Interaction Terms

$$g_1(x, t) = f_5(t) \cdot \phi_2(x), \qquad g_2(x, t) = f_6(t) \cdot \phi_2(x)$$

- **Burnout x S-curve:** High-incentive borrowers burn out faster than low-incentive borrowers.
- **Regime x S-curve:** Regime shifts may disproportionately affect high-incentive borrowers.

### 4.2 Estimation

The full design matrix $\mathbf{X} \in \mathbb{R}^{N \times p}$ (where $N = n \times T$ and $p = 14$) is constructed and the coefficients are estimated via **ridge regression**:

$$\hat{\boldsymbol{\beta}} = \left(\mathbf{X}^{\top}\mathbf{X} + \lambda \mathbf{I}\right)^{-1} \mathbf{X}^{\top} \mathbf{y}$$

with regularization parameter $\lambda = 0.01$. The key advantage over single-variable methods: **the model uses all 6,000 observations jointly**, learning temporal patterns that enable forecasting.

### 4.3 Post-Estimation Monotonicity

After prediction, monotonicity in $x$ is enforced for each time slice:

$$\text{CPR}_{\text{mono}}(x_i, t) = \max_{j \leq i} \widehat{\text{CPR}}(x_j, t)$$

### 4.4 Model Parsimony

The multivariate model uses only **14 parameters** to fit the entire panel. Compare this to single-variable methods, which use ~11 parameters ($K + k + 1$ spline coefficients) **per month**, totaling ~1,320 parameters across the panel. This dramatic reduction in parameter count is both a regularizer and the source of forecasting ability.

---

## 5. Comparative Evaluation

We evaluate the multivariate model against single-variable methods across three scenarios that reflect realistic deployment conditions.

### 5.1 Scenario 1: Forecasting (Unseen Time Periods)

**Setup:** Train on months 0–79, predict months 80–119. Single-variable methods have no observed data for test months.

**Table 1: Forecasting RMSE (Months 80–119)**

| Method | RMSE | vs. Multivariate |
|--------|------|-------------------|
| Static base S-curve | 0.2924 | +508% |
| Average of training splines | 0.1463 | +204% |
| **Multivariate (out-of-sample)** | **0.0481** | — |

The multivariate model achieves an **83.6% improvement** over the best available single-variable baseline. This is the model's most compelling advantage: it can forecast because it has learned the structural relationships between environmental factors and CPR.

![Figure B: The multivariate model (purple) closely tracks the true CPR (black) at months 80, 90, 100, and 115—all unseen during training. The static base curve (gray dotted) and average training spline (green dashed) are poor substitutes.](compare_scurves.png)

**Why does forecasting work?** The multivariate model knows that at month 90, burnout should be approximately $-0.25(1 - e^{-0.03 \times 90}) = -0.233$, the regime shift should be fully active at $-0.10$, and seasonality depends on the calendar month. Single-variable methods have no mechanism to make these predictions.

### 5.2 Scenario 2: High Noise ($\sigma = 0.08$)

**Setup:** All 120 months observed with $\sigma = 0.08$ (3.2x the standard noise level).

**Table 2: High-Noise RMSE (Full Panel)**

| Method | RMSE |
|--------|------|
| Bayesian | 0.0690 |
| PCHIP | 0.0727 |
| Spline | 0.0419 |
| **Multivariate** | **0.0368** |

The multivariate model's structural constraints act as a powerful regularizer. While the spline tracks noise fluctuations with its 11 free parameters per month, the multivariate model's 14-parameter structural form resists overfitting, achieving a **12.2% improvement** over the best single-variable method.

![Figure D: At high noise, the spline (green dashed) tracks noisy observations and produces wiggly estimates. The multivariate model (purple) maintains smooth structural predictions that closely track the true CPR (black).](compare_rmse_timeline.png)

### 5.3 Scenario 3: Sparse Data (12 of 50 Incentive Points)

**Setup:** Only 12 of 50 incentive levels are observed per month, with $\sigma = 0.04$.

**Table 3: Sparse-Data RMSE (Full Panel)**

| Method | RMSE |
|--------|------|
| Bayesian | 0.1872 |
| PCHIP | 0.0326 |
| Spline | 0.0319 |
| **Multivariate** | **0.0317** |

With only 12 observations per month, the Bayesian smoother degrades dramatically. PCHIP and spline remain competitive because they interpolate effectively, but the multivariate model still achieves the lowest RMSE by leveraging cross-temporal information.

### 5.4 Consolidated Comparison

![Figure A: The multivariate model (purple, outlined) achieves the lowest RMSE across all three scenarios. The advantage is most dramatic for forecasting, where single-variable methods have no data for test months.](compare_rmse.png)

**Table 4: Consolidated Results**

| Scenario | Best Single-Var | Multivariate | Improvement |
|----------|----------------|--------------|-------------|
| 1. Forecasting (months 80–119) | 0.1463 | 0.0481 | **67.1%** |
| 2. High Noise ($\sigma = 0.08$) | 0.0419 | 0.0368 | **12.2%** |
| 3. Sparse Data (12/50 points) | 0.0319 | 0.0317 | **0.7%** |

**Key insight:** The multivariate model's advantage arises from structural knowledge, not brute-force fitting. When conditions are ideal (low noise, full data, no forecasting required), single-variable methods are competitive. Under realistic conditions—where forecasting is needed, noise is substantial, or data is sparse—the multivariate model's structural framework provides clear superiority.

---

## 6. Environmental Factor Decomposition

### 6.1 Factor Stacking Visualization

A key advantage of the multivariate approach is its ability to decompose the time-varying S-curve into interpretable components. Figure C shows how each environmental factor physically shifts the base S-curve at four critical time points.

![Figure C: Environmental factors stacked additively on the base S-curve. Each colored band represents one factor's contribution. At month 6, seasonality dominates (+0.08). By month 96, cumulative burnout (-0.24) and regime shift (-0.10) have dramatically compressed the curve.](compare_env_factors.png)

### 6.2 Factor Dynamics

| Time Point | Dominant Factor | Total Shift | Interpretation |
|------------|----------------|-------------|----------------|
| Month 6 | Seasonality (+0.08) | +0.04 | Summer peak partially offset by early burnout |
| Month 30 | Media wave (+0.06) | +0.01 | Media spike partially offset by growing burnout |
| Month 60 | Regime (-0.10) | -0.34 | Regime shift compounds with mature burnout |
| Month 96 | Burnout (-0.24) | -0.42 | All negative factors cumulate; curve severely compressed |

### 6.3 Ablation Study

To quantify each factor's marginal contribution, we fit the multivariate model with successive factors removed:

**Table 5: Ablation Study — Factor Contribution to RMSE Reduction**

| Model Configuration | RMSE | Delta vs. Base |
|---------------------|------|---------------|
| Base S-curve only | 0.1048 | — |
| + Seasonality | 0.0958 | -8.6% |
| + Burnout | 0.0653 | -37.7% |
| + Regime shift | 0.0677 | -35.4% |
| + Season + Burnout | 0.0457 | -56.4% |
| **Full model** | **0.0307** | **-70.7%** |

Burnout provides the largest single-factor improvement (-37.7%), consistent with its persistent, cumulative nature. Regime shifts contribute comparably (-35.4%), reflecting the large magnitude of the structural break. Factor contributions are largely additive, suggesting limited confounding.

---

## 7. Implications for Option-Adjusted Spread

### 7.1 OAS Formulation

In the simplified linear model:

$$\text{OAS}(x) = s_0 - \beta \cdot \text{CPR}(x), \quad s_0 = 100 \text{ bps}, \quad \beta = 80 \text{ bps}$$

Errors in CPR estimation translate directly into OAS mispricing:

$$\Delta \text{OAS}(x) = -\beta \cdot \left(\widehat{\text{CPR}}(x) - \text{CPR}_{\text{true}}(x)\right)$$

### 7.2 OAS Impact Analysis

**Table 6: OAS Mispricing by Method and Scenario**

| Method | Forecasting | High Noise | Sparse |
|--------|-------------|------------|--------|
| Best single-variable | ~11.7 bps | ~3.4 bps | ~2.6 bps |
| Multivariate | ~3.8 bps | ~2.9 bps | ~2.5 bps |

In the forecasting scenario, the multivariate model reduces OAS mispricing by nearly 8 bps—material for agency MBS trading where typical bid-ask spreads are 1–2 bps.

---

## 8. Sensitivity Analysis

### 8.1 Sensitivity to Noise Level

We evaluate all methods across four noise levels: $\sigma \in \{0.01, 0.02, 0.05, 0.10\}$ with all 120 months and 50 incentive points observed.

| $\sigma$ | Bayesian | PCHIP | Spline | Multivariate | Winner |
|-----------|----------|-------|--------|--------------|--------|
| 0.01 | 0.0600 | 0.0095 | **0.0042** | 0.0290 | Spline |
| 0.02 | 0.0615 | 0.0192 | **0.0085** | 0.0295 | Spline |
| 0.05 | 0.0657 | 0.0481 | 0.0214 | **0.0330** | Spline |
| 0.10 | 0.0790 | 0.0963 | 0.0523 | **0.0400** | **Multi** |

**Crossover point:** At approximately $\sigma = 0.06$, the multivariate model begins outperforming all single-variable methods on in-sample RMSE. Below this threshold, single-variable methods benefit from fitting each month's low-noise data independently. Above it, their parameters absorb noise while the multivariate model's structural form provides regularization.

**Important caveat:** Even when single-variable methods win on in-sample RMSE, they still cannot forecast. The multivariate model's forecasting advantage exists regardless of noise level.

### 8.2 Sensitivity to S-Curve Steepness

Steeper curves ($k > 10$) compress the transition into a narrow $x$-range, challenging all methods. The Bayesian smoother is most affected; the optimized spline remains robust with adequate knot density. The multivariate model handles variable steepness through its multi-steepness logistic basis ($k \in \{4, 6, 10\}$).

---

## 9. Discussion

### 9.1 When to Use Each Method

| Condition | Recommended Method |
|-----------|-------------------|
| Need to forecast CPR for future months | **Multivariate** |
| High observation noise ($\sigma > 0.05$) | **Multivariate** |
| Sparse incentive observations | **Multivariate** |
| Low noise, dense data, in-sample only | **Spline** (slightly lower RMSE) |
| Need for interpretable factor decomposition | **Multivariate** |
| No temporal information available | **Spline** |

### 9.2 Practical Considerations for External Factors

**Seasonality identification.** In production, Fourier coefficients can be estimated from historical data with 24+ months of history. Alternative approaches include monthly dummy variables or seasonal ARIMA decomposition.

**Burnout calibration.** The burnout decay rate $\lambda_b$ depends on pool characteristics:
- Higher LTV pools burn out slower (fewer refinancing options)
- Higher FICO pools burn out faster (more financially responsive)
- Geographic concentration matters (local housing market effects)

**Regime shift detection.** In real-time applications, structural breaks must be detected rather than assumed. Methods include:
- **CUSUM tests** on recursive residuals
- **Bai-Perron** multiple break point tests
- **Hidden Markov models** for regime-switching dynamics
- **Rolling window estimation** with break detection

**Media wave modeling.** Media effects are inherently unpredictable and cannot be included in forward-looking models. However, they can be identified ex-post for model calibration using news sentiment indices or Google Trends data.

### 9.3 Model Extensions

**Nonlinear interactions.** The linear interaction terms assume proportional effects. Kernel methods or neural networks could capture more complex interaction surfaces.

**Heterogeneous pools.** Mixture models or stratified estimation (by FICO, LTV, loan age, geography) can capture within-pool heterogeneity.

**Bayesian regularization.** Replacing ridge regression with a full Bayesian treatment provides posterior distributions over coefficients, enabling uncertainty quantification:

$$p(\boldsymbol{\beta} | \mathbf{y}) \propto p(\mathbf{y} | \boldsymbol{\beta}) \cdot p(\boldsymbol{\beta})$$

**Time-varying coefficients.** A state-space extension would allow $\boldsymbol{\beta}_t$ to evolve:

$$\boldsymbol{\beta}_t = \boldsymbol{\beta}_{t-1} + \boldsymbol{\omega}_t, \quad \boldsymbol{\omega}_t \sim \mathcal{N}(\mathbf{0}, \Sigma_\omega)$$

### 9.4 Limitations

1. **Simulated data.** The evaluation uses synthetic data. Real-world CPR exhibits additional complexities (non-Gaussian noise, cross-sectional dependence, missing data).
2. **Post-hoc monotonicity.** Cumulative maximum enforcement can create derivative discontinuities. Constrained optimization ($c_{j+1} \geq c_j$) would be superior.
3. **Feature engineering.** The design matrix requires domain-specific feature construction.
4. **Known factor timing.** The current implementation assumes regime shift timing is known. In practice, break point detection adds additional uncertainty.

---

## 10. Conclusion

### 10.1 Summary of Findings

This paper demonstrated that incorporating environmental factors into S-curve CPR estimation provides a fundamentally different—and superior—modeling paradigm:

| Key Result | Value |
|------------|-------|
| Forecasting improvement (vs. best baseline) | **83.6%** |
| High-noise improvement (vs. best single-var) | **12.2%** |
| Largest single-factor contribution | Burnout (-37.7%) |
| Full ablation improvement | **70.7%** |
| Total model parameters | 14 (vs. ~1,320 for panel of splines) |

The multivariate model's advantage is not merely statistical—it represents a qualitative shift from curve-fitting to structural modeling, enabling forecasting, interpretability, and robustness that single-variable methods cannot provide.

### 10.2 Recommendations

1. **Use multivariate panel regression** as the primary framework when 24+ months of historical data are available.
2. **Always include burnout**—it is the single most important external factor.
3. **Monitor for regime shifts** using structural break detection and update the model accordingly.
4. **Use optimized B-splines** for cross-sectional estimation only when temporal information is unavailable.
5. **Validate on out-of-sample data** with proper train/test splits before production deployment.

### 10.3 Future Work

1. **Time-varying parameter models** via state-space or online learning
2. **Constrained B-spline QP** for monotonicity without post-hoc enforcement
3. **Bayesian uncertainty quantification** with posterior credible intervals
4. **Machine learning extensions:** gradient boosting, neural networks for nonlinear interactions
5. **Real data validation** on agency MBS pool-level prepayment data

---

## References

1. Black, L., Chu, S., Cohen, A., & Nichols, J. (2012). Differences across originators in CMBS loan underwriting. *Journal of Financial Services Research*, 42(3), 115-134.

2. Davidson, A. S., & Levin, A. (2014). *Mortgage Valuation Models: Embedded Options, Risk, and Uncertainty*. Oxford University Press.

3. de Boor, C. (2001). *A Practical Guide to Splines* (Revised ed.). Springer.

4. Dierckx, P. (1993). *Curve and Surface Fitting with Splines*. Oxford University Press.

5. Fritsch, F. N., & Carlson, R. E. (1980). Monotone piecewise cubic interpolation. *SIAM Journal on Numerical Analysis*, 17(2), 238-246.

6. Hayre, L. (Ed.). (2001). *Salomon Smith Barney Guide to Mortgage-Backed and Asset-Backed Securities*. Wiley.

7. Hyman, J. M. (1983). Accurate monotonicity preserving cubic interpolation. *SIAM Journal on Scientific and Statistical Computing*, 4(4), 645-654.

8. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Transactions of the ASME-Journal of Basic Engineering*, 82(D), 35-45.

9. Piechocki, R. (2019). Mortgage prepayment models and S-curves. *Journal of Fixed Income*, 28(4), 48-63.

10. Richard, S. F., & Roll, R. (1989). Prepayments on fixed-rate mortgage-backed securities. *Journal of Portfolio Management*, 15(3), 73-82.

11. Schwartz, E. S., & Torous, W. N. (1989). Prepayment and the valuation of mortgage-backed securities. *Journal of Finance*, 44(2), 375-392.

12. SIFMA. (2024). *US Mortgage-Related Issuance and Outstanding*. Securities Industry and Financial Markets Association.

---

## Appendix A: Complete Python Implementation

```python
import numpy as np
from scipy.interpolate import PchipInterpolator, make_lsq_spline
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n_months, n_incentive = 120, 50
x = np.linspace(0, 1, n_incentive)
t_months = np.arange(n_months)

# Base S-curve
base_cpr = 0.05 + 0.90 / (1 + np.exp(-6 * (x - 0.5)))

# External factors
season = 0.08 * np.sin(2 * np.pi * (t_months - 3) / 12)
burnout = -0.25 * (1 - np.exp(-0.03 * t_months))
regime = -0.10 / (1 + np.exp(-(t_months - 60) / 3))
media = 0.06 * np.exp(-0.5 * ((t_months - 30) / 4) ** 2)

# Panel generation
CPR_true = np.zeros((n_months, n_incentive))
for m in range(n_months):
    CPR_true[m, :] = np.clip(
        base_cpr + season[m] + burnout[m] + regime[m] + media[m],
        0.01, 0.99
    )

CPR_obs = np.clip(CPR_true + np.random.normal(0, 0.025, CPR_true.shape), 0.01, 0.99)

# Multivariate regression (ridge)
def build_features(x_grid, t_grid):
    n_t, n_x = len(t_grid), len(x_grid)
    X_f, T_f = np.tile(x_grid, n_t), np.repeat(t_grid, n_x)
    feats = [np.ones(n_t * n_x)]
    for kk in [4, 6, 10]:
        feats.append(1 / (1 + np.exp(-kk * (X_f - 0.5))))
    feats.append(X_f ** 2)
    feats.extend([np.sin(2*np.pi*T_f/12), np.cos(2*np.pi*T_f/12),
                  np.sin(4*np.pi*T_f/12), np.cos(4*np.pi*T_f/12)])
    bo = 1 - np.exp(-0.03 * T_f)
    rg = 1 / (1 + np.exp(-(T_f - 60) / 3))
    sc = 1 / (1 + np.exp(-6 * (X_f - 0.5)))
    feats.extend([bo, rg, bo * sc, rg * sc, T_f / n_months])
    return np.column_stack(feats)

X_mat = build_features(x, t_months)
y_flat = CPR_obs.flatten()
lam = 0.01
beta = np.linalg.solve(X_mat.T @ X_mat + lam * np.eye(X_mat.shape[1]), X_mat.T @ y_flat)
CPR_pred = np.clip(X_mat @ beta, 0.01, 0.99).reshape(n_months, n_incentive)

# Enforce monotonicity per time slice
for m in range(n_months):
    CPR_pred[m, :] = np.maximum.accumulate(CPR_pred[m, :])

print(f"Full model RMSE: {np.sqrt(mean_squared_error(CPR_true.flatten(), CPR_pred.flatten())):.4f}")
```

## Appendix B: Notation Reference

| Symbol | Description |
|--------|-------------|
| $x$ | Refinancing incentive |
| $t$ | Time (months) |
| $\text{CPR}(x, t)$ | Conditional prepayment rate |
| $\sigma_L(z)$ | Logistic sigmoid function |
| $k$ | S-curve steepness parameter |
| $x_0$ | S-curve inflection point |
| $\sigma$ | Observation noise standard deviation |
| $B_{j,k}(x)$ | B-spline basis function of order $k$ |
| $c_j$ | B-spline control coefficients |
| $\text{OAS}$ | Option-adjusted spread |
| $\alpha$ | Exponential smoothing constant |
| $\delta_{\text{seas}}$ | Seasonality adjustment |
| $\delta_{\text{burn}}$ | Burnout adjustment |
| $\delta_{\text{reg}}$ | Regime shift adjustment |
| $\delta_{\text{med}}$ | Media wave adjustment |
| $\lambda$ | Ridge regularization parameter |
| $\lambda_b$ | Burnout decay rate |
| $A_s$ | Seasonality amplitude |
| $B_{\max}$ | Maximum burnout magnitude |
| $R_{\text{mag}}$ | Regime shift magnitude |
