import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler



def permutation_significance(model, X_test_scaled, y_test_scaled,baseline_r2, feature_cols, n_permutations=1000):


    y_test_flat = y_test_scaled.ravel()

    # Baseline R² on real unshuffled test data
    #baseline_r2 = r2_score(y_test_flat, model.predict(X_test_scaled))
    print(f"\nBaseline R²: {baseline_r2:.4f}")
    print(f"Running {n_permutations} permutations per feature...\n")

    results = []

    for f_idx, feature in enumerate(feature_cols):

        null_r2_scores = np.zeros(n_permutations)

        for _ in range(n_permutations):
            X_permuted = X_test_scaled.copy()

            # Shuffle only this feature column → break its relationship with y
            X_permuted[:, f_idx] = np.random.permutation(X_permuted[:, f_idx])

            null_r2_scores[_] = r2_score(y_test_flat, model.predict(X_permuted))

        # R² DROP = how much performance falls when feature is shuffled
        r2_drop = baseline_r2 - null_r2_scores.mean()

        # p-value = how often null R² >= baseline (i.e. shuffling didn't hurt)
        #p_value = np.mean(null_r2_scores >= baseline_r2)
        p_value = (np.sum(null_r2_scores >= baseline_r2) + 1) / (n_permutations + 1)

        results.append({
            "feature": feature,
            "baseline_r2": baseline_r2,
            "mean_null_r2": null_r2_scores.mean(),
            "r2_drop": r2_drop,
            "std_null_r2": null_r2_scores.std(),
            "p_value": p_value,
            "significant (p<0.05)": p_value < 0.05,
            # Bonferroni corrected for 11 features
            "significant (bonferroni)": p_value < (0.05 / len(feature_cols))
        })

        print(f"{feature:35s} | R² drop: {r2_drop:+.4f} | p = {p_value:.9f}")

    results_df = pd.DataFrame(results).sort_values("r2_drop", ascending=False)
    return results_df, null_r2_scores


# =========================================================
# Plot Null Distribution for Each Feature
# =========================================================

# def plot_null_distributions(model, X_test_scaled, y_test_scaled, baseline_r2, feature_cols, n_permutations=1000):
#     """
#     Plot null R² distribution vs baseline for each feature.
#     """
#
#     y_test_flat = y_test_scaled.ravel()
#     #baseline_r2 = r2_score(y_test_flat, model.predict(X_test_scaled))
#
#     n_features = len(feature_cols)
#     ncols = 3
#     nrows = int(np.ceil(n_features / ncols))
#
#     fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
#     axes = axes.flatten()
#
#     for f_idx, feature in enumerate(feature_cols):
#
#         null_r2_scores = np.zeros(n_permutations)
#
#         for _ in range(n_permutations):
#             X_permuted = X_test_scaled.copy()
#             X_permuted[:, f_idx] = np.random.permutation(X_permuted[:, f_idx])
#             null_r2_scores[_] = r2_score(y_test_flat, model.predict(X_permuted))
#
#         p_value = np.mean(null_r2_scores >= baseline_r2)
#
#         ax = axes[f_idx]
#         ax.hist(null_r2_scores, bins=40, color="steelblue", alpha=0.7, label="Null R²")
#         ax.axvline(baseline_r2, color="red", linestyle="--", linewidth=2, label=f"Baseline R²={baseline_r2:.3f}")
#         ax.axvline(null_r2_scores.mean(), color="orange", linestyle="--", linewidth=1.5,
#                    label=f"Null mean={null_r2_scores.mean():.3f}")
#         ax.set_title(f"{feature}\np = {p_value:.4f}", fontsize=10)
#         ax.set_xlabel("R²")
#         ax.set_ylabel("Count")
#         ax.legend(fontsize=7)
#
#     # Hide unused subplots
#     for j in range(n_features, len(axes)):
#         axes[j].set_visible(False)
#
#     plt.suptitle("Permutation Test — Null R² Distributions per Feature", fontsize=13, fontweight="bold")
#     plt.tight_layout()
#     plt.savefig("permutation_null_distributions.png", dpi=300, bbox_inches="tight")
#     plt.show()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    feature_cols = [
        "h3.3_gene_body_ct12",
        "h3.3_promoters_ct12",
        "h3k27ac_ct12",
        "h3k9ac_ct12",
        "h3k4me3_ct12",
        "h3k4me1_ct12",
        "h3k36me3_gene_body_ct12",
        "h3k79me2_gene_body_ct12",
        "per1_promoters_ct12",
        "per2_promoters_ct12",
        "rnapol2_promoter_ct12"
    ]

    # =========================================================
    # Load Data
    # =========================================================

    data = pd.read_csv(
        'rnapol2_8wg16.csv',
        na_values=["NA", "null", "?", " "],
        engine='python'
    )

    print(data.columns)

    X = data[feature_cols].values
    y = data["ct12_rpkm_cm_avg"].values.reshape(-1, 1)

    # =========================================================
    # Train / Test Split + Scaling
    # =========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=999
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_scaled = np.log1p(y_train)
    y_test_scaled = np.log1p(y_test)

    # =========================================================
    # Train Model
    # =========================================================

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=2,
        max_depth=None,
        random_state=999
    )

    model.fit(X_train_scaled, y_train_scaled.ravel())

    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test_scaled, y_pred)
    mae = mean_absolute_error(y_test_scaled, y_pred)

    print(f"R²:  {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # =========================================================
    # Save Predictions
    # =========================================================

    pd.DataFrame({
        "y_actual": y_test_scaled.ravel(),
        "y_pred": y_pred.ravel()
    }).to_csv("y_test.csv", index=False)

    # =========================================================
    # Permutation Significance (replaces SHAP)
    # =========================================================

    results_df, _ = permutation_significance(
        model,
        X_test_scaled,
        y_test_scaled,
        r2,
        feature_cols,
        n_permutations=1000
    )

    results_df.to_csv("permutation_significance.csv", index=False)

    print("\n========== PERMUTATION TEST RESULTS ==========")
    print(results_df.to_string(index=False))

    # =========================================================
    # Plot Null Distributions
    # =========================================================

    # plot_null_distributions(
    #     model,
    #     X_test_scaled,
    #     y_test_scaled,
    #     r2,
    #     feature_cols,
    #     n_permutations=1000
    # )

    # =========================================================
    # Bar Plot — R² Drop per Feature
    # =========================================================

    plt.figure(figsize=(10, 6))
    colors = [
        "red" if s else "steelblue"
        for s in results_df["significant (bonferroni)"]
    ]
    plt.barh(results_df["feature"], results_df["r2_drop"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("R² Drop (higher = more important)")
    plt.title("Permutation Feature Importance\n(red = significant after Bonferroni correction)")
    plt.tight_layout()
    plt.savefig("permutation_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.show()
