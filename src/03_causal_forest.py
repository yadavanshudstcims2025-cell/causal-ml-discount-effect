# ============================================================
# STEP 3: CAUSAL FOREST
# ============================================================
#
# Objective:
# Estimate:
#
# 1. Average Treatment Effect (ATE)
# 2. Individual Treatment Effects (CATE)
# 3. Accuracy of estimated CATE compared with TRUE CATE
#
# Step 2 showed:
# Naive ATE = 0.0209
# True ATE  = 0.0150
#
# Now we use Causal Forest to account for customer
# characteristics and estimate heterogeneous treatment effects.
# ============================================================

import pandas as pd
import numpy as np

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

data = pd.read_csv("data/synthetic_data.csv")


# ------------------------------------------------------------
# 2. Define X, T and Y
# ------------------------------------------------------------

feature_columns = [
    "previous_purchases",
    "avg_order_value",
    "days_since_last_purchase",
    "email_clicked"
]

# Customer characteristics
X = data[feature_columns]

# Treatment
# 1 = discount
# 0 = no discount
T = data["treatment"].astype(float)

# Observed outcome
# 1 = purchase
# 0 = no purchase
Y = data["outcome"].astype(float)


# ------------------------------------------------------------
# 3. TRUE treatment effects
# ------------------------------------------------------------
#
# This is available because we created synthetic data.
#
# In real-world data, these values would NOT be observed.
# ------------------------------------------------------------

true_cate = data["true_individual_effect"].values

true_ate = true_cate.mean()


# ------------------------------------------------------------
# 4. Create Causal Forest DML
# ------------------------------------------------------------

print("Training Causal Forest...")

cf = CausalForestDML(

    # Outcome model
    model_y=RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    ),

    # Treatment model
    model_t=RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        random_state=42
    ),

    # Number of causal trees
    n_estimators=300,

    # Minimum observations per leaf
    min_samples_leaf=10,

    # Maximum tree depth
    max_depth=5,

    random_state=42
)


# ------------------------------------------------------------
# 5. Fit the Causal Forest
# ------------------------------------------------------------

cf.fit(
    Y,
    T,
    X=X
)


# ------------------------------------------------------------
# 6. Estimate individual treatment effects
# ------------------------------------------------------------

estimated_cate = cf.effect(X)


# ------------------------------------------------------------
# 7. Estimate ATE
# ------------------------------------------------------------

estimated_ate = estimated_cate.mean()


# ------------------------------------------------------------
# 8. ATE error
# ------------------------------------------------------------

ate_error = (
    estimated_ate - true_ate
)

ate_absolute_error = abs(ate_error)

ate_relative_error = (
    ate_absolute_error /
    abs(true_ate)
) * 100


# ------------------------------------------------------------
# 9. Naive estimator for comparison
# ------------------------------------------------------------

naive_ate = (
    data.loc[
        data["treatment"] == 1,
        "outcome"
    ].mean()
    -
    data.loc[
        data["treatment"] == 0,
        "outcome"
    ].mean()
)

naive_absolute_error = abs(
    naive_ate - true_ate
)

naive_relative_error = (
    naive_absolute_error /
    abs(true_ate)
) * 100


# ------------------------------------------------------------
# 10. CATE RMSE
# ------------------------------------------------------------
#
# Compare estimated individual treatment effects against
# the TRUE individual treatment effects.
# ------------------------------------------------------------

cate_rmse = np.sqrt(
    mean_squared_error(
        true_cate,
        estimated_cate
    )
)


# ------------------------------------------------------------
# 11. Correlation between TRUE and estimated CATE
# ------------------------------------------------------------

cate_correlation = np.corrcoef(
    true_cate,
    estimated_cate
)[0, 1]


# ------------------------------------------------------------
# 12. Add estimated CATE to dataset
# ------------------------------------------------------------

data["estimated_cate"] = estimated_cate


# ------------------------------------------------------------
# 13. Calculate CATE error for every customer
# ------------------------------------------------------------

data["cate_error"] = (
    data["estimated_cate"]
    -
    data["true_individual_effect"]
)


# ============================================================
# 14. PRINT ATE RESULTS
# ============================================================

print("=" * 65)
print("STEP 3: CAUSAL FOREST RESULTS")
print("=" * 65)

print(
    f"True ATE:              "
    f"{true_ate:.4f}"
)

print(
    f"Naive ATE:              "
    f"{naive_ate:.4f}"
)

print(
    f"Causal Forest ATE:     "
    f"{estimated_ate:.4f}"
)

print("-" * 65)

print(
    f"Naive absolute error:   "
    f"{naive_absolute_error:.4f}"
)

print(
    f"Causal Forest absolute error: "
    f"{ate_absolute_error:.4f}"
)

print("-" * 65)

print(
    f"Naive relative error:   "
    f"{naive_relative_error:.2f}%"
)

print(
    f"Causal Forest relative error: "
    f"{ate_relative_error:.2f}%"
)

print("=" * 65)


# ============================================================
# 15. HETEROGENEOUS EFFECT RESULTS
# ============================================================

print("\nHETEROGENEOUS TREATMENT EFFECTS")
print("=" * 65)

print(
    f"True minimum effect: "
    f"{true_cate.min():.4f}"
)

print(
    f"True maximum effect: "
    f"{true_cate.max():.4f}"
)

print(
    f"Estimated minimum effect: "
    f"{estimated_cate.min():.4f}"
)

print(
    f"Estimated maximum effect: "
    f"{estimated_cate.max():.4f}"
)

print("-" * 65)

print(
    f"CATE RMSE: "
    f"{cate_rmse:.4f}"
)

print(
    f"True vs estimated CATE correlation: "
    f"{cate_correlation:.4f}"
)


# ============================================================
# 16. TOP CUSTOMERS
# ============================================================

print("\nTOP 10 CUSTOMERS BY ESTIMATED TREATMENT EFFECT")
print("=" * 65)

top_customers = (
    data
    .sort_values(
        "estimated_cate",
        ascending=False
    )
    .head(10)
)

print(
    top_customers[
        feature_columns
        +
        [
            "true_individual_effect",
            "estimated_cate"
        ]
    ].to_string(index=False)
)


# ============================================================
# 17. BUSINESS INTERPRETATION
# ============================================================

print("\nBUSINESS INTERPRETATION")
print("=" * 65)

if ate_absolute_error < naive_absolute_error:

    print(
        "Causal Forest produced an ATE closer to the "
        "true ATE than the naive estimator."
    )

else:

    print(
        "Causal Forest did not produce an ATE closer "
        "to the true ATE than the naive estimator."
    )


if cate_correlation > 0:

    print(
        "The positive CATE correlation indicates that "
        "the model is capturing some of the differences "
        "in treatment response across customers."
    )

else:

    print(
        "The CATE correlation is not positive, indicating "
        "that the model is not reliably identifying "
        "individual treatment-effect differences."
    )

print(
    "These estimated individual effects can be used "
    "in Step 4 to target customers who are predicted "
    "to benefit most from discounts."
)
