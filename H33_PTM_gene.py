import os

import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV

import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
#from cuml.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, TweedieRegressor, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from os import scandir
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.svm import SVR
import seaborn as sns
import matplotlib as mpl
import shap
import xgboost as xgb




def shap_explain(model, X_test):
    features = [
        "h33_gb",
        "h33_pm",
        "k27ac",
        "k9ac",
        "k4me3",
        "k4me1",
        "k36me3_gb",
        "k79me2_gb",
        "per1_promoters",
        "per2_promoters",
        "rnapol2_promoter"
    ]


    X_test = pd.DataFrame(X_test, columns=features)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.plots.beeswarm(shap_values)
    #plt.savefig("/home/group_nithya01/Dharani/ml/location-specific/dataset2/shap/h33_ptm_rna_location_specific_2_ct0_shap.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
       #[df_ML_X, df_ML_Y]= load_csv()
       data = pd.read_csv('rnapol2_8wg16.csv',na_values=["NA", "null", "?", " "],engine='python')
       #data = pd.read_csv('h33_ptm_location_specific_2_per_rna_megadepth_single_strand_ct0_latest.csv', na_values=["NA", "null", "?", " "],engine='python')

       print(data.columns)

       h33_ct12_gb = data["h3.3_gene_body_ct12"].values
       h33_ct12_pm = data["h3.3_promoters_ct12"].values
       k27ac_ct12=data["h3k27ac_ct12"].values
       k9ac_ct12=data["h3k9ac_ct12"].values
       k4me3_ct12=data["h3k4me3_ct12"].values
       k4me1_ct12 = data["h3k4me1_ct12"].values
       k36me3_gb_ct12=data["h3k36me3_gene_body_ct12"].values
       k79me2_gb_ct12=data["h3k79me2_gene_body_ct12"].values
       rnapol2_promoter_ct12= data["rnapol2_promoter_ct12"].values
       per2_promoters_ct12= data["per2_promoters_ct12"].values
       per1_promoters_ct12= data["per1_promoters_ct12"].values

       #per1_ct12 = data["per1_ct12"].values
       #per2_ct12 = data["per2_ct12"].values

       #
       #X=np.concatenate([h33_ct12_gb.reshape(-1,1),h33_ct12_pm.reshape(-1,1),k27ac_ct12.reshape(-1,1),k9ac_ct12.reshape(-1,1), k4me3_ct12.reshape(-1,1), k4me1_ct12.reshape(-1,1),k36me3_gb_ct12.reshape(-1,1),k79me2_gb_ct12.reshape(-1,1),per1_ct12.reshape(-1,1),per2_ct12.reshape(-1,1)],axis=1)
       X=np.concatenate([h33_ct12_gb.reshape(-1,1),h33_ct12_pm.reshape(-1,1),k27ac_ct12.reshape(-1,1),k9ac_ct12.reshape(-1,1), k4me3_ct12.reshape(-1,1), k4me1_ct12.reshape(-1,1),k36me3_gb_ct12.reshape(-1,1),k79me2_gb_ct12.reshape(-1,1),per1_promoters_ct12.reshape(-1,1),per2_promoters_ct12.reshape(-1,1),rnapol2_promoter_ct12.reshape(-1,1)],axis=1)

       #
       #y=data["rnaseq_0"].values.reshape(-1,1)
       y=data["ct12_rpkm_cm_avg"].values.reshape(-1,1)

       #

       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=999)
       # #
       scalar_X= RobustScaler()
       X_train_scaled=scalar_X.fit_transform(X_train)
       X_test_scaled=scalar_X.transform(X_test)

       #scalar_Y= StandardScaler()
       y_train_scaled = np.log1p(y_train)
       y_test_scaled = np.log1p(y_test)

       #y_train_scaled = (y_train)
       #y_test_scaled = (y_test)



       #
       #model = SVR(kernel='rbf')
       #model=LinearRegression()
       # model=xgb.XGBRegressor(  n_estimators=500,
       #  learning_rate=0.05,
       #  max_depth=None,
       #  subsample=0.8,
       #  colsample_bytree=0.8,
       #  random_state=999)
       model=RandomForestRegressor(n_estimators=200,min_samples_split=2,max_depth=None)
       model.fit(X_train_scaled, y_train_scaled)
       y_pred = model.predict(X_test_scaled)

       #y_pred = np.expm1(y_pred)
       #y_test_orig = np.expm1(y_test_scaled)

       r2 = r2_score(y_test_scaled, y_pred)
       mae = mean_absolute_error(y_test_scaled, y_pred)
       print("R2:",r2)
       mae=mean_absolute_error(y_test_scaled,y_pred)
       print("MAE:",mae)

       y_test_df = pd.DataFrame({
           "y_test_scaled": y_test_scaled.ravel(),
           "y_pred": y_pred.ravel(),

       })
       y_test_df.to_csv("y_test.csv")

       print("Variance:",np.var(y_test_scaled))
       print("Min:",np.min(y_test_scaled))
       print("Max:",np.max(y_test_scaled))
       plt.scatter(y_test_scaled, y_pred)
       plt.xlabel("Y actual")
       plt.ylabel("Y predicted")
       plt.xlim(0,10)
       plt.ylim(0,10)
       plt.show()

       #kf = KFold(n_splits=5, shuffle=True, random_state=42)
       # #
       #scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
       # #
       # print(scores)
       # print("Mean R2 across 5 folds ", scores.mean())
       # print("Std of R2 across 5 folds ", scores.std())
       #
       idx = np.random.permutation(X_test_scaled.shape[0])
       X_shuffled = X_test_scaled[idx]

       shap_explain(model,X_shuffled)




