import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import shap
import xgboost as xgb
from sklearn.pipeline import Pipeline


from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    Lasso,
    LassoCV,
    LinearRegression,
    TweedieRegressor,
    Ridge
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
    RobustScaler
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.model_selection import (
    LeaveOneOut, cross_val_predict
)

from sklearn.base import clone


def shap_explain(model, X_test):

    features = [
        "h33_ct0_gb",
        "h33_ct0_pm",
        "k27ac_ct0",
        "k9ac_ct0",
        "k4me3_ct0",
        "k4me1_ct0",
        "k36me3_gb_ct0",
        "k79me2_gb_ct0"
    ]

    X_test = pd.DataFrame(X_test, columns=features)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values)


if __name__ == "__main__":

    data = pd.read_csv(
        'rnapol2_8wg16.csv',
        na_values=["NA", "null", "?", " "],
        engine='python'
    )

    print(data.columns)

    h33_ct12_gb = data["h3.3_gene_body_ct12"].values
    h33_ct12_pm = data["h3.3_promoters_ct12"].values
    k27ac_ct12 = data["h3k27ac_ct12"].values
    k9ac_ct12 = data["h3k9ac_ct12"].values
    k4me3_ct12 = data["h3k4me3_ct12"].values
    k4me1_ct12 = data["h3k4me1_ct12"].values
    k36me3_gb_ct12 = data["h3k36me3_gene_body_ct12"].values
    k79me2_gb_ct12 = data["h3k79me2_gene_body_ct12"].values
    rnapol2_promoter_ct12 = data["rnapol2_promoter_ct12"].values
    per2_promoters_ct12 = data["per2_promoters_ct12"].values
    per1_promoters_ct12 = data["per1_promoters_ct12"].values

    X = np.concatenate([
        h33_ct12_gb.reshape(-1, 1),
        h33_ct12_pm.reshape(-1, 1),
        k27ac_ct12.reshape(-1, 1),
        k9ac_ct12.reshape(-1, 1),
        k4me3_ct12.reshape(-1, 1),
        k4me1_ct12.reshape(-1, 1),
        k36me3_gb_ct12.reshape(-1, 1),
        k79me2_gb_ct12.reshape(-1, 1),
        per1_promoters_ct12.reshape(-1, 1),
        per2_promoters_ct12.reshape(-1, 1),
        rnapol2_promoter_ct12.reshape(-1, 1)
    ], axis=1)

    y = data["ct12_rpkm_cm_avg"].values.reshape(-1, 1)

    # =========================================================
    # LOOCV
    # =========================================================

    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', RandomForestRegressor(
            n_estimators=200,
            min_samples_split=2,
            max_depth=None,
            random_state=999
        ))
    ])

    # =========================================================
    # LOOCV Predictions
    # =========================================================

    loo = LeaveOneOut()

    y_pred_all = cross_val_predict(pipeline, X, y, cv=loo)

    # =========================================================
    # Evaluation
    # =========================================================

    r2 = r2_score(y, y_pred_all)
    mae = mean_absolute_error(y, y_pred_all)
    mse = mean_squared_error(y, y_pred_all)

    print(f"R2:  {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")

    with open("output.txt", "w") as f:
       
        f.write(f"R2:  {r2:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"MSE: {mse:.4f}\n")

    # =========================================================
    # Save Predictions
    # =========================================================

    pd.DataFrame({
        "y_actual": y,
        "y_pred": y_pred_all
    }).to_csv("loocv_predictions.csv", index=False)

    # =========================================================
    # Stats
    # =========================================================

    print("\nVariance:", np.var(y))
    print("Min:     ", np.min(y))
    print("Max:     ", np.max(y))

