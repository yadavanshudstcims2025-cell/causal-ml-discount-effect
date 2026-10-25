# ============================================================
# STEP 5: STATISTICAL VALIDATION
# ============================================================
#
# This step contains:
#
# 1. Bootstrap confidence interval for ATE
# 2. Hypothesis testing
# 3. Placebo test
# 4. Final visualizations
#
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.linear_model import LinearRegression


# ============================================================
# 1. Create output directory
# ============================================================

os.makedirs(
    "results/plots",
    exist_ok=True
)


# ============================================================
# 2. Load data
# ============================================================

data = pd.read_csv(
    "data/synthetic_data.csv"
)


# ============================================================
# 3. Define variables
# ============================================================

feature_columns = [
    "previous_purchases",
    "avg_order_value",
    "days_since_last_purchase",
    "email_clicked"
]

X = data[feature_columns]

T = data["treatment"].values

Y = data["outcome"].values


# ============================================================
# 4. NAIVE ATE
# ============================================================

treated_mean = Y[T == 1].mean()

control_mean = Y[T == 0].mean()

naive_ate = (
    treated_mean -
    control_mean
)


# ============================================================
# 5. TRUE ATE
# ============================================================

true_ate = data[
    "true_individual_effect"
].mean()


print("=" * 65)
print("STEP 5: STATISTICAL VALIDATION")
print("=" * 65)

print(
    f"Naive ATE: {naive_ate:.4f}"
)

print(
    f"True ATE:  {true_ate:.4f}"
)


# ============================================================
# PART A: BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================
#
# We repeatedly resample customers with replacement.
#
# For every bootstrap sample:
#
# ATE = mean(Y | T=1)
#       -
#       mean(Y | T=0)
#
# Then we use the 2.5th and 97.5th percentiles
# to construct a 95% bootstrap confidence interval.
# ============================================================

print("\n" + "=" * 65)
print("PART A: BOOTSTRAP CONFIDENCE INTERVAL")
print("=" * 65)


n_bootstrap = 500

bootstrap_ates = []

rng = np.random.default_rng(42)


for i in range(n_bootstrap):

    # Randomly sample customers with replacement
    sample_indices = rng.choice(
        len(data),
        size=len(data),
        replace=True
    )

    sample = data.iloc[
        sample_indices
    ]


    # Make sure both treatment groups exist
    if (
        sample["treatment"].nunique()
        < 2
    ):
        continue


    treated = sample[
        sample["treatment"] == 1
    ]["outcome"].mean()


    control = sample[
        sample["treatment"] == 0
    ]["outcome"].mean()


    bootstrap_ate = (
        treated -
        control
    )


    bootstrap_ates.append(
        bootstrap_ate
    )


bootstrap_ates = np.array(
    bootstrap_ates
)


# ============================================================
# Calculate 95% confidence interval
# ============================================================

lower_ci = np.percentile(
    bootstrap_ates,
    2.5
)

upper_ci = np.percentile(
    bootstrap_ates,
    97.5
)


bootstrap_mean = (
    bootstrap_ates.mean()
)


print(
    f"Bootstrap replications: "
    f"{len(bootstrap_ates)}"
)

print(
    f"Bootstrap mean ATE: "
    f"{bootstrap_mean:.4f}"
)

print(
    f"95% Bootstrap CI: "
    f"[{lower_ci:.4f}, {upper_ci:.4f}]"
)


# ============================================================
# Interpretation
# ============================================================

if (
    lower_ci <= 0
    and upper_ci >= 0
):

    print(
        "The confidence interval contains zero."
    )

else:

    print(
        "The confidence interval does not contain zero."
    )


# ============================================================
# Bootstrap visualization
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    bootstrap_ates,
    bins=40,
    edgecolor="black"
)

plt.axvline(
    lower_ci,
    linestyle="--",
    linewidth=2,
    label=f"2.5% = {lower_ci:.4f}"
)

plt.axvline(
    upper_ci,
    linestyle="--",
    linewidth=2,
    label=f"97.5% = {upper_ci:.4f}"
)

plt.axvline(
    true_ate,
    linestyle="-",
    linewidth=2,
    label=f"True ATE = {true_ate:.4f}"
)

plt.xlabel(
    "Bootstrap ATE"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Bootstrap Distribution of ATE"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/plots/07_bootstrap_ate.png",
    dpi=300
)

plt.close()


# ============================================================
# PART B: HYPOTHESIS TEST
# ============================================================
#
# H0: ATE = 0
#
# H1: ATE != 0
#
# We use a two-sample test for the observed purchase rates.
# ============================================================

print("\n" + "=" * 65)
print("PART B: HYPOTHESIS TEST")
print("=" * 65)


treated_outcomes = data[
    data["treatment"] == 1
]["outcome"]

control_outcomes = data[
    data["treatment"] == 0
]["outcome"]


# Welch's t-test
t_statistic, p_value = stats.ttest_ind(
    treated_outcomes,
    control_outcomes,
    equal_var=False
)


print(
    f"T-statistic: "
    f"{t_statistic:.4f}"
)

print(
    f"P-value: "
    f"{p_value:.6f}"
)


alpha = 0.05


if p_value < alpha:

    print(
        "Decision: Reject H0 at the 5% significance level."
    )

    print(
        "There is statistically significant evidence "
        "of a difference in purchase rates."
    )

else:

    print(
        "Decision: Fail to reject H0 at the 5% significance level."
    )

    print(
        "There is insufficient evidence of a difference "
        "in purchase rates."
    )


# ============================================================
# PART C: PLACEBO TEST
# ============================================================
#
# We randomly shuffle the treatment assignment.
#
# This destroys the actual relationship between treatment
# and outcome.
#
# Therefore, the placebo treatment should produce an
# estimated effect close to zero.
# ============================================================

print("\n" + "=" * 65)
print("PART C: PLACEBO TEST")
print("=" * 65)


n_placebo = 500

placebo_ates = []


for i in range(n_placebo):

    # Randomly shuffle treatment
    placebo_T = rng.permutation(T)


    # Calculate placebo ATE
    placebo_treated = Y[
        placebo_T == 1
    ].mean()


    placebo_control = Y[
        placebo_T == 0
    ].mean()


    placebo_ate = (
        placebo_treated -
        placebo_control
    )


    placebo_ates.append(
        placebo_ate
    )


placebo_ates = np.array(
    placebo_ates
)


# ============================================================
# Placebo results
# ============================================================

placebo_mean = (
    placebo_ates.mean()
)

placebo_lower = np.percentile(
    placebo_ates,
    2.5
)

placebo_upper = np.percentile(
    placebo_ates,
    97.5
)


print(
    f"Mean placebo ATE: "
    f"{placebo_mean:.4f}"
)

print(
    f"95% placebo interval: "
    f"[{placebo_lower:.4f}, "
    f"{placebo_upper:.4f}]"
)


# ============================================================
# Placebo visualization
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    placebo_ates,
    bins=40,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero effect"
)

plt.axvline(
    naive_ate,
    linestyle="-",
    linewidth=2,
    label=f"Observed ATE = {naive_ate:.4f}"
)

plt.xlabel(
    "Placebo ATE"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Placebo Treatment Effect Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/plots/08_placebo_test.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("STEP 5 SUMMARY")
print("=" * 65)

print(
    f"True ATE:              {true_ate:.4f}"
)

print(
    f"Naive ATE:             {naive_ate:.4f}"
)

print(
    f"Bootstrap mean ATE:    {bootstrap_mean:.4f}"
)

print(
    f"95% Bootstrap CI:      "
    f"[{lower_ci:.4f}, {upper_ci:.4f}]"
)

print(
    f"Hypothesis test p-value: "
    f"{p_value:.6f}"
)

print(
    f"Mean placebo ATE:      "
    f"{placebo_mean:.4f}"
)

print(
    f"95% placebo interval:  "
    f"[{placebo_lower:.4f}, "
    f"{placebo_upper:.4f}]"
)

print("=" * 65)

print(
    "\nPlots saved to:"
)

print(
    "results/plots/07_bootstrap_ate.png"
)

print(
    "results/plots/08_placebo_test.png"
)

print("=" * 65)
