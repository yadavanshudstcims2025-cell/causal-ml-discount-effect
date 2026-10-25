# ============================================================
# STEP 2: NAIVE ESTIMATION
# ============================================================

import pandas as pd
import numpy as np

# Load the NEW data generated in Step 1
data = pd.read_csv("data/synthetic_data.csv")


# ------------------------------------------------------------
# 1. Separate treatment groups
# ------------------------------------------------------------

treated = data[data["treatment"] == 1]

control = data[data["treatment"] == 0]


# ------------------------------------------------------------
# 2. Calculate observed purchase rates
# ------------------------------------------------------------

treated_mean = treated["outcome"].mean()

control_mean = control["outcome"].mean()


# ------------------------------------------------------------
# 3. Naive ATE
# ------------------------------------------------------------

naive_ate = (
    treated_mean
    - control_mean
)


# ------------------------------------------------------------
# 4. TRUE ATE
# ------------------------------------------------------------
#
# Unlike the previous version, we don't hard-code 0.015.
# We calculate it from the true individual effects created
# in Step 1.
# ------------------------------------------------------------

true_ate = (
    data["true_individual_effect"].mean()
)


# ------------------------------------------------------------
# 5. Calculate bias
# ------------------------------------------------------------

bias = naive_ate - true_ate

absolute_error = abs(bias)

relative_error = (
    absolute_error / abs(true_ate)
) * 100


# ------------------------------------------------------------
# 6. Confounding check
# ------------------------------------------------------------

treated_aov = treated["avg_order_value"].mean()

control_aov = control["avg_order_value"].mean()


# ------------------------------------------------------------
# 7. Print results
# ------------------------------------------------------------

print("=" * 65)
print("STEP 2: NAIVE ESTIMATION")
print("=" * 65)

print(
    f"Treated customers: "
    f"{len(treated)}"
)

print(
    f"Control customers: "
    f"{len(control)}"
)

print("-" * 65)

print(
    f"Treated purchase rate: "
    f"{treated_mean:.4f}"
)

print(
    f"Control purchase rate: "
    f"{control_mean:.4f}"
)

print("-" * 65)

print(
    f"Naive ATE: "
    f"{naive_ate:.4f}"
)

print(
    f"True ATE: "
    f"{true_ate:.4f}"
)

print(
    f"Bias: "
    f"{bias:.4f}"
)

print(
    f"Absolute error: "
    f"{absolute_error:.4f}"
)

print(
    f"Relative error: "
    f"{relative_error:.2f}%"
)

print("-" * 65)

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


# ------------------------------------------------------------
# 8. Interpretation
# ------------------------------------------------------------

print("\nINTERPRETATION")
print("=" * 65)

print(
    "The naive estimator compares the observed purchase "
    "rates of discounted and non-discounted customers."
)

print(
    "Because high-value customers are more likely to receive "
    "discounts, treatment assignment is confounded."
)

print(
    f"The naive estimate is {naive_ate:.4f}, "
    f"while the true ATE is {true_ate:.4f}."
)

if naive_ate > true_ate:
    print(
        "Therefore, the naive method overestimates the "
        "average causal effect of the discount."
    )
else:
    print(
        "Therefore, the naive method underestimates the "
        "average causal effect of the discount."
    )
