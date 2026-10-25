# ============================================================
# STEP 6: ROI & TARGETING STRATEGY
# ============================================================
#
# Objective:
# Translate causal treatment effects into business value.
#
# We compare:
# 1. Top 10% uplift targeting
# 2. Top 30% uplift targeting
# 3. Random 30% targeting
#
# BUSINESS ASSUMPTIONS
# --------------------
# Discount rate = 10%
#
# True individual treatment effect represents the incremental
# probability of purchase caused by the discount.
#
# Incremental purchases:
#       sum(true treatment effect)
#
# Incremental revenue:
#       sum(true treatment effect * average order value)
#
# Discount cost:
#       incremental purchases * average order value
#       * discount rate
#
# ROI:
#       (Incremental Revenue - Discount Cost)
#       / Discount Cost * 100
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Create output directory
# ============================================================

os.makedirs(
    "results/plots",
    exist_ok=True
)


# ============================================================
# 2. Load Step 4 results
# ============================================================

data = pd.read_csv(
    "results/uplift_targeting_results.csv"
)


# ============================================================
# 3. Business assumptions
# ============================================================

DISCOUNT_RATE = 0.10

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)


print("=" * 65)
print("STEP 6: ROI & TARGETING STRATEGY")
print("=" * 65)

print("\nBUSINESS ASSUMPTIONS")
print("-" * 65)

print(
    f"Discount rate: "
    f"{DISCOUNT_RATE:.0%}"
)

print(
    "Revenue per incremental purchase = "
    "Average Order Value"
)

print(
    "Discount cost is calculated only on "
    "incremental purchases."
)


# ============================================================
# 4. ROI calculation function
# ============================================================

def calculate_roi(customers):

    # --------------------------------------------------------
    # Incremental purchases
    # --------------------------------------------------------
    #
    # true_individual_effect is the increase in purchase
    # probability caused by the discount.
    #
    # Summing these probabilities gives expected incremental
    # purchases.
    # --------------------------------------------------------

    incremental_purchases = (
        customers[
            "true_individual_effect"
        ].sum()
    )


    # --------------------------------------------------------
    # Incremental revenue
    # --------------------------------------------------------
    #
    # For each customer:
    #
    # incremental probability
    # × average order value
    # --------------------------------------------------------

    incremental_revenue = (
        customers[
            "true_individual_effect"
        ]
        *
        customers[
            "avg_order_value"
        ]
    ).sum()


    # --------------------------------------------------------
    # Discount cost
    # --------------------------------------------------------
    #
    # Cost of the discount on incremental purchases only.
    # --------------------------------------------------------

    discount_cost = (
        customers[
            "true_individual_effect"
        ]
        *
        customers[
            "avg_order_value"
        ]
        *
        DISCOUNT_RATE
    ).sum()


    # --------------------------------------------------------
    # Net incremental value
    # --------------------------------------------------------

    net_incremental_value = (
        incremental_revenue
        -
        discount_cost
    )


    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    if discount_cost > 0:

        roi = (
            net_incremental_value
            /
            discount_cost
        ) * 100

    else:

        roi = np.nan


    return {
        "customers":
            len(customers),

        "incremental_purchases":
            incremental_purchases,

        "incremental_revenue":
            incremental_revenue,

        "discount_cost":
            discount_cost,

        "net_incremental_value":
            net_incremental_value,

        "ROI_percent":
            roi
    }


# ============================================================
# 5. Rank customers using predicted uplift
# ============================================================

data = data.sort_values(
    "predicted_uplift",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 6. TOP 10% TARGETING
# ============================================================

n_10 = int(
    0.10 * len(data)
)

top_10 = data.iloc[
    :n_10
]

result_10 = calculate_roi(
    top_10
)


# ============================================================
# 7. TOP 30% TARGETING
# ============================================================

n_30 = int(
    0.30 * len(data)
)

top_30 = data.iloc[
    :n_30
]

result_30 = calculate_roi(
    top_30
)


# ============================================================
# 8. RANDOM 30% TARGETING
# ============================================================

random_30 = data.sample(
    n=n_30,
    random_state=RANDOM_SEED
)

result_random = calculate_roi(
    random_30
)


# ============================================================
# 9. Create ROI comparison table
# ============================================================

roi_results = pd.DataFrame([
    {
        "strategy":
            "Top 10% Uplift Targeting",

        **result_10
    },

    {
        "strategy":
            "Top 30% Uplift Targeting",

        **result_30
    },

    {
        "strategy":
            "Random 30% Targeting",

        **result_random
    }
])


# ============================================================
# 10. Print ROI results
# ============================================================

print("\n" + "=" * 65)
print("ROI COMPARISON")
print("=" * 65)

print(
    roi_results.to_string(
        index=False,
        formatters={

            "incremental_purchases":
                "{:.2f}".format,

            "incremental_revenue":
                "{:.2f}".format,

            "discount_cost":
                "{:.2f}".format,

            "net_incremental_value":
                "{:.2f}".format,

            "ROI_percent":
                "{:.2f}%".format
        }
    )
)


# ============================================================
# 11. Compare Top 30% vs Random 30%
# ============================================================

top30_roi = (
    result_30[
        "ROI_percent"
    ]
)

random30_roi = (
    result_random[
        "ROI_percent"
    ]
)


roi_gain = (
    top30_roi -
    random30_roi
)


print("\n" + "=" * 65)
print("TARGETING STRATEGY COMPARISON")
print("=" * 65)

print(
    f"Top 30% uplift ROI: "
    f"{top30_roi:.2f}%"
)

print(
    f"Random 30% ROI: "
    f"{random30_roi:.2f}%"
)

print(
    f"ROI improvement: "
    f"{roi_gain:.2f} percentage points"
)


# ============================================================
# 12. Relative improvement
# ============================================================

if random30_roi != 0:

    relative_improvement = (
        (
            top30_roi -
            random30_roi
        )
        /
        abs(random30_roi)
    ) * 100

else:

    relative_improvement = np.nan


print(
    f"Relative ROI improvement: "
    f"{relative_improvement:.2f}%"
)


# ============================================================
# 13. Business interpretation
# ============================================================

print("\n" + "=" * 65)
print("BUSINESS INTERPRETATION")
print("=" * 65)


if top30_roi > random30_roi:

    print(
        "The uplift-based Top 30% strategy "
        "outperforms random targeting."
    )

    print(
        "This indicates that the uplift model "
        "provides some value for discount targeting."
    )

else:

    print(
        "The uplift-based Top 30% strategy "
        "does not outperform random targeting."
    )

    print(
        "The current model does not provide enough "
        "individual-level targeting value."
    )


# ============================================================
# 14. Additional business metrics
# ============================================================

top30_incremental_revenue = (
    result_30[
        "incremental_revenue"
    ]
)

random30_incremental_revenue = (
    result_random[
        "incremental_revenue"
    ]
)

revenue_gain = (
    top30_incremental_revenue
    -
    random30_incremental_revenue
)


print("\n" + "=" * 65)
print("INCREMENTAL REVENUE COMPARISON")
print("=" * 65)

print(
    f"Top 30% incremental revenue: "
    f"{top30_incremental_revenue:.2f}"
)

print(
    f"Random 30% incremental revenue: "
    f"{random30_incremental_revenue:.2f}"
)

print(
    f"Revenue gain from uplift targeting: "
    f"{revenue_gain:.2f}"
)


# ============================================================
# 15. ROI visualization
# ============================================================

plt.figure(
    figsize=(9, 6)
)

strategies = roi_results[
    "strategy"
]

roi_values = roi_results[
    "ROI_percent"
]

plt.bar(
    strategies,
    roi_values,
    edgecolor="black"
)

plt.ylabel(
    "ROI (%)"
)

plt.xlabel(
    "Targeting Strategy"
)

plt.title(
    "ROI Comparison of Discount Targeting Strategies"
)

plt.xticks(
    rotation=15
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "results/plots/09_roi_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. Incremental revenue visualization
# ============================================================

plt.figure(
    figsize=(9, 6)
)

revenue_values = [
    result_10[
        "incremental_revenue"
    ],

    result_30[
        "incremental_revenue"
    ],

    result_random[
        "incremental_revenue"
    ]
]

plt.bar(
    strategies,
    revenue_values,
    edgecolor="black"
)

plt.ylabel(
    "Expected Incremental Revenue"
)

plt.xlabel(
    "Targeting Strategy"
)

plt.title(
    "Expected Incremental Revenue by Targeting Strategy"
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.savefig(
    "results/plots/10_incremental_revenue.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. Save ROI results
# ============================================================

roi_results.to_csv(
    "results/roi_targeting_results.csv",
    index=False
)


# ============================================================
# 18. Final output
# ============================================================

print("\n" + "=" * 65)
print("STEP 6 COMPLETED")
print("=" * 65)

print(
    "ROI results saved to:"
)

print(
    "results/roi_targeting_results.csv"
)

print("\nPlots saved to:")

print(
    "results/plots/09_roi_comparison.png"
)

print(
    "results/plots/10_incremental_revenue.png"
)

print("=" * 65)
