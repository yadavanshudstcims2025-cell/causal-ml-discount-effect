# ============================================================
# STEP 4: CROSS-FITTED TWO-MODEL UPLIFT MODELING
# ============================================================
#
# Objective:
# Identify customers who are most likely to benefit from
# receiving a discount.
#
# We use two Random Forest models:
#
# Model 1:
# P(Y=1 | X, Discount)
#
# Model 2:
# P(Y=1 | X, No Discount)
#
# Predicted Uplift =
#
# P(Y=1 | X, Discount)
# -
# P(Y=1 | X, No Discount)
#
# IMPORTANT:
# We use out-of-fold predictions so that each customer's
# prediction is produced by a model that did NOT train
# directly on that customer's outcome.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. Create output folders
# ============================================================

os.makedirs("results/plots", exist_ok=True)


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

true_effect = data[
    "true_individual_effect"
].values


# ============================================================
# 4. Create arrays for out-of-fold predictions
# ============================================================

predicted_purchase_if_treated = np.zeros(
    len(data)
)

predicted_purchase_if_control = np.zeros(
    len(data)
)


# ============================================================
# 5. Five-fold cross-fitting
# ============================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("=" * 65)
print("STEP 4: CROSS-FITTED TWO-MODEL UPLIFT MODEL")
print("=" * 65)

print("\nTraining 5-fold uplift models...")


for fold, (train_index, test_index) in enumerate(
    skf.split(X, T),
    start=1
):

    print(f"Processing fold {fold}/5...")


    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train = X.iloc[train_index]

    T_train = T[train_index]

    Y_train = Y[train_index]


    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    X_test = X.iloc[test_index]


    # --------------------------------------------------------
    # Treatment training data
    # --------------------------------------------------------

    treatment_mask = (
        T_train == 1
    )

    X_treatment = X_train[
        treatment_mask
    ]

    Y_treatment = Y_train[
        treatment_mask
    ]


    # --------------------------------------------------------
    # Control training data
    # --------------------------------------------------------

    control_mask = (
        T_train == 0
    )

    X_control = X_train[
        control_mask
    ]

    Y_control = Y_train[
        control_mask
    ]


    # ========================================================
    # MODEL 1: TREATMENT MODEL
    # ========================================================

    treatment_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=30,
        random_state=42 + fold,
        n_jobs=-1
    )

    treatment_model.fit(
        X_treatment,
        Y_treatment
    )


    # ========================================================
    # MODEL 2: CONTROL MODEL
    # ========================================================

    control_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=30,
        random_state=100 + fold,
        n_jobs=-1
    )

    control_model.fit(
        X_control,
        Y_control
    )


    # ========================================================
    # Predict on validation fold
    # ========================================================

    predicted_purchase_if_treated[
        test_index
    ] = treatment_model.predict_proba(
        X_test
    )[:, 1]


    predicted_purchase_if_control[
        test_index
    ] = control_model.predict_proba(
        X_test
    )[:, 1]


# ============================================================
# 6. Calculate cross-fitted uplift
# ============================================================

predicted_uplift = (
    predicted_purchase_if_treated
    -
    predicted_purchase_if_control
)


# Add predictions to dataset

data[
    "purchase_if_discount"
] = predicted_purchase_if_treated


data[
    "purchase_if_no_discount"
] = predicted_purchase_if_control


data[
    "predicted_uplift"
] = predicted_uplift


# ============================================================
# 7. Overall uplift performance
# ============================================================

uplift_rmse = np.sqrt(
    mean_squared_error(
        true_effect,
        predicted_uplift
    )
)

uplift_correlation = np.corrcoef(
    true_effect,
    predicted_uplift
)[0, 1]


print("\n" + "=" * 65)
print("UPLIFT MODEL RESULTS")
print("=" * 65)

print(
    f"Mean predicted uplift: "
    f"{predicted_uplift.mean():.4f}"
)

print(
    f"Mean true treatment effect: "
    f"{true_effect.mean():.4f}"
)

print("-" * 65)

print(
    f"Uplift RMSE: "
    f"{uplift_rmse:.4f}"
)

print(
    f"True vs predicted uplift correlation: "
    f"{uplift_correlation:.4f}"
)


# ============================================================
# 8. Rank customers
# ============================================================

data = data.sort_values(
    "predicted_uplift",
    ascending=False
).reset_index(
    drop=True
)


data[
    "target_rank"
] = np.arange(
    1,
    len(data) + 1
)


# ============================================================
# 9. Create targeting percentiles
# ============================================================
#
# We examine:
#
# Top 10%
# Top 20%
# Top 30%
# Top 40%
# ...
# Top 100%
#
# For each group we calculate the TRUE average treatment
# effect.
# ============================================================

percentages = [
    10, 20, 30, 40, 50,
    60, 70, 80, 90, 100
]


targeting_results = []


for percentage in percentages:

    n_target = int(
        len(data) * percentage / 100
    )

    selected = data.iloc[
        :n_target
    ]


    mean_true_effect = (
        selected[
            "true_individual_effect"
        ].mean()
    )


    mean_predicted_uplift = (
        selected[
            "predicted_uplift"
        ].mean()
    )


    targeting_results.append({

        "percentage_targeted":
            percentage,

        "customers_targeted":
            n_target,

        "mean_predicted_uplift":
            mean_predicted_uplift,

        "mean_true_effect":
            mean_true_effect
    })


targeting_results = pd.DataFrame(
    targeting_results
)


# ============================================================
# 10. Print targeting performance
# ============================================================

print("\n" + "=" * 65)
print("TARGETING PERFORMANCE")
print("=" * 65)

print(
    targeting_results.to_string(
        index=False
    )
)


# ============================================================
# 11. Top 20 customers
# ============================================================

print("\n" + "=" * 65)
print("TOP 20 CUSTOMERS FOR DISCOUNT TARGETING")
print("=" * 65)

print(
    data[
        [
            "target_rank",
            "previous_purchases",
            "avg_order_value",
            "days_since_last_purchase",
            "email_clicked",
            "true_individual_effect",
            "predicted_uplift"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 12. Business targeting groups
# ============================================================

top_10 = data.iloc[
    :int(0.10 * len(data))
]

top_30 = data.iloc[
    :int(0.30 * len(data))
]

remaining = data.iloc[
    int(0.30 * len(data)) :
]


print("\n" + "=" * 65)
print("BUSINESS TARGETING COMPARISON")
print("=" * 65)

print(
    f"Top 10% true average effect: "
    f"{top_10['true_individual_effect'].mean():.4f}"
)

print(
    f"Top 30% true average effect: "
    f"{top_30['true_individual_effect'].mean():.4f}"
)

print(
    f"Remaining 70% true average effect: "
    f"{remaining['true_individual_effect'].mean():.4f}"
)


# ============================================================
# 13. Calculate targeting lift
# ============================================================

overall_true_ate = (
    data[
        "true_individual_effect"
    ].mean()
)

top_10_true_effect = (
    top_10[
        "true_individual_effect"
    ].mean()
)

targeting_lift = (
    top_10_true_effect
    -
    overall_true_ate
)


print(
    f"\nTop 10% targeting lift over "
    f"overall ATE: {targeting_lift:.4f}"
)


# ============================================================
# 14. Visualization 1
#     True vs predicted uplift
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    true_effect,
    predicted_uplift,
    alpha=0.30
)

minimum = min(
    true_effect.min(),
    predicted_uplift.min()
)

maximum = max(
    true_effect.max(),
    predicted_uplift.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "True Individual Treatment Effect"
)

plt.ylabel(
    "Predicted Uplift"
)

plt.title(
    "True vs Predicted Individual Treatment Effect"
)

plt.tight_layout()

plt.savefig(
    "results/plots/04_true_vs_predicted_uplift.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. Visualization 2
#     Targeting curve
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    targeting_results[
        "percentage_targeted"
    ],
    targeting_results[
        "mean_true_effect"
    ],
    marker="o"
)

plt.axhline(
    y=overall_true_ate,
    linestyle="--",
    label="Overall True ATE"
)

plt.xlabel(
    "Percentage of Customers Targeted"
)

plt.ylabel(
    "Average True Treatment Effect"
)

plt.title(
    "Uplift Targeting Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/plots/05_uplift_targeting_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. Visualization 3
#     Predicted uplift distribution
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.hist(
    predicted_uplift,
    bins=40,
    edgecolor="black"
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Uplift"
)

plt.ylabel(
    "Number of Customers"
)

plt.title(
    "Distribution of Predicted Customer Uplift"
)

plt.tight_layout()

plt.savefig(
    "results/plots/06_uplift_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. Save targeting results
# ============================================================

data[
    feature_columns
    + [
        "treatment",
        "outcome",
        "true_individual_effect",
        "purchase_if_discount",
        "purchase_if_no_discount",
        "predicted_uplift",
        "target_rank"
    ]
].to_csv(
    "results/uplift_targeting_results.csv",
    index=False
)


targeting_results.to_csv(
    "results/targeting_performance.csv",
    index=False
)


# ============================================================
# 18. Final interpretation
# ============================================================

print("\n" + "=" * 65)
print("STEP 4 COMPLETED")
print("=" * 65)

if (
    top_10_true_effect
    > overall_true_ate
):

    print(
        "The top 10% targeted customers have a "
        "higher average true treatment effect than "
        "the overall population."
    )

else:

    print(
        "The top 10% targeted customers do not have "
        "a higher average true treatment effect than "
        "the overall population."
    )


print(
    "\nTargeting results saved to:"
)

print(
    "results/uplift_targeting_results.csv"
)

print(
    "results/targeting_performance.csv"
)

print("\nPlots saved to:")

print(
    "results/plots/04_true_vs_predicted_uplift.png"
)

print(
    "results/plots/05_uplift_targeting_curve.png"
)

print(
    "results/plots/06_uplift_distribution.png"
)

print("=" * 65)
