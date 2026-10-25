# ============================================================
# STEP 1: GENERATE SYNTHETIC CAUSAL DATA
# ============================================================
#
# Objective:
# Create a realistic observational customer dataset in which:
#
# 1. Discount assignment is CONFOUNDED.
# 2. Customers have different characteristics.
# 3. The TRUE treatment effect varies across customers.
# 4. The average TRUE treatment effect is approximately 0.015.
#
# This allows us to study:
#       - Naive estimation
#       - Causal Forest
#       - Uplift modeling
#       - Statistical validation
# ============================================================

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------

np.random.seed(42)


# ------------------------------------------------------------
# 2. Number of customers
# ------------------------------------------------------------

n_customers = 10_000

# Create data directory
os.makedirs("data", exist_ok=True)


# ------------------------------------------------------------
# 3. Generate customer characteristics
# ------------------------------------------------------------
#
# These variables represent information a company might
# observe about its customers.
# ------------------------------------------------------------

data = pd.DataFrame({

    # Number of purchases made previously
    "previous_purchases":
        np.random.randint(0, 50, n_customers),

    # Typical amount spent per order
    "avg_order_value":
        np.random.uniform(10, 500, n_customers),

    # Number of days since the customer last purchased
    "days_since_last_purchase":
        np.random.randint(1, 365, n_customers),

    # Whether the customer clicked a marketing email
    # Approximately 35% are assumed to click
    "email_clicked":
        np.random.binomial(1, 0.35, n_customers)
})


# ============================================================
# 4. CONFONDED TREATMENT ASSIGNMENT
# ============================================================
#
# Treatment:
#     1 = customer receives discount
#     0 = customer does not receive discount
#
# Assumption:
# High-value customers are more likely to receive discounts.
#
# Therefore:
#
# Average order value
#        ↓
#   Discount assignment
#
# and also:
#
# Average order value
#        ↓
#   Purchase probability
#
# This creates CONFOUNDING.
# ============================================================

prob_treatment = (
    0.20
    + 0.50 * (
        data["avg_order_value"] / 500
    )
)

# Ensure probabilities remain between 0 and 1
prob_treatment = np.clip(
    prob_treatment,
    0.05,
    0.90
)

# Randomly assign treatment according to probability
data["treatment"] = (
    np.random.random(n_customers)
    < prob_treatment
).astype(int)


# ============================================================
# 5. BASELINE PURCHASE PROBABILITY
# ============================================================
#
# This represents the probability of purchase WITHOUT
# receiving a discount.
#
# Customer characteristics affect this probability.
# ============================================================

baseline_prob = (
    0.10
    + 0.002 * data["previous_purchases"]
    + 0.00015 * data["avg_order_value"]
    + 0.05 * data["email_clicked"]
)


# Keep probabilities within a valid range
baseline_prob = np.clip(
    baseline_prob,
    0.01,
    0.80
)


# ============================================================
# 6. CREATE HETEROGENEOUS TRUE TREATMENT EFFECT
# ============================================================
#
# IMPORTANT DIFFERENCE FROM THE PREVIOUS VERSION:
#
# Previously:
#
#     treatment_effect = 0.015
#
# for EVERY customer.
#
# Now each customer gets a slightly different treatment
# effect.
#
# The effect depends on customer characteristics.
#
# This allows Causal Forest and uplift modeling to study
# heterogeneous treatment effects.
# ============================================================

raw_effect = (
    0.005

    + 0.00015 * data["previous_purchases"]

    + 0.000015 * data["avg_order_value"]

    + 0.002 * data["email_clicked"]
)


# ------------------------------------------------------------
# 7. Scale effects so that the AVERAGE true effect = 0.015
# ------------------------------------------------------------
#
# We want the overall average treatment effect to remain
# approximately 0.015.
#
# This means:
#
#     Average TRUE individual effect ≈ 0.015
#
# while individual effects differ.
# ------------------------------------------------------------

target_ate = 0.015

scaling_factor = (
    target_ate / raw_effect.mean()
)

true_individual_effect = (
    raw_effect * scaling_factor
)


# ------------------------------------------------------------
# 8. Store the TRUE individual treatment effect
# ------------------------------------------------------------
#
# This column is normally unavailable in real-world data.
# We keep it because this is a simulation and it allows us
# to evaluate whether our ML methods recover the truth.
# ------------------------------------------------------------

data["true_individual_effect"] = (
    true_individual_effect
)


# ============================================================
# 9. CREATE POTENTIAL OUTCOME PROBABILITIES
# ============================================================
#
# Y(0):
# Purchase probability WITHOUT discount
#
# Y(1):
# Purchase probability WITH discount
#
# Causal effect:
#
#     Y(1) - Y(0)
#
# = true_individual_effect
# ============================================================

prob_without_discount = baseline_prob

prob_with_discount = (
    baseline_prob
    + true_individual_effect
)


# Keep probabilities valid
prob_with_discount = np.clip(
    prob_with_discount,
    0.01,
    0.95
)


# ============================================================
# 10. CREATE OBSERVED PURCHASE OUTCOME
# ============================================================
#
# We only observe ONE potential outcome for each customer.
#
# If treatment = 1:
#     observe Y(1)
#
# If treatment = 0:
#     observe Y(0)
# ============================================================

observed_probability = np.where(
    data["treatment"] == 1,
    prob_with_discount,
    prob_without_discount
)


# Generate binary purchase outcome
data["outcome"] = (
    np.random.random(n_customers)
    < observed_probability
).astype(int)


# ============================================================
# 11. SAVE DATA
# ============================================================

file_path = "data/synthetic_data.csv"

data.to_csv(
    file_path,
    index=False
)


# ============================================================
# 12. CALCULATE TRUE ATE
# ============================================================
#
# Because we created the true individual effects ourselves,
# we can calculate the actual average causal effect.
# ============================================================

true_ate = (
    data["true_individual_effect"].mean()
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("=" * 65)
print("STEP 1: SYNTHETIC CAUSAL DATA GENERATED")
print("=" * 65)

print(
    f"Number of customers: "
    f"{n_customers}"
)

print(
    f"Treatment rate: "
    f"{data['treatment'].mean():.2%}"
)

print(
    f"Overall purchase rate: "
    f"{data['outcome'].mean():.2%}"
)

print("-" * 65)

print(
    f"Purchase rate - No discount: "
    f"{data.loc[data['treatment'] == 0, 'outcome'].mean():.2%}"
)

print(
    f"Purchase rate - Discount: "
    f"{data.loc[data['treatment'] == 1, 'outcome'].mean():.2%}"
)

print("-" * 65)

print(
    f"TRUE ATE: "
    f"{true_ate:.4f}"
)

print(
    f"TRUE ATE in percentage points: "
    f"{true_ate * 100:.2f}%"
)

print("-" * 65)

print(
    f"Minimum true individual effect: "
    f"{data['true_individual_effect'].min():.4f}"
)

print(
    f"Maximum true individual effect: "
    f"{data['true_individual_effect'].max():.4f}"
)

print(
    f"Median true individual effect: "
    f"{data['true_individual_effect'].median():.4f}"
)

print("-" * 65)

# Confounding check
treated_aov = data.loc[
    data["treatment"] == 1,
    "avg_order_value"
].mean()

control_aov = data.loc[
    data["treatment"] == 0,
    "avg_order_value"
].mean()

print("CONFOUNDING CHECK")

print(
    f"Average order value - Treated: "
    f"{treated_aov:.2f}"
)

print(
    f"Average order value - Control: "
    f"{control_aov:.2f}"
)

print("=" * 65)

print(
    f"Data saved to: {file_path}"
)
