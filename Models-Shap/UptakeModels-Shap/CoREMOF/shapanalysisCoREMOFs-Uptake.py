# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:03:11 2023

@author: user
"""
import os 
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import shap
from tpot.export_utils import set_param_recursive
import matplotlib.pyplot as plt 

plt.rcParams["font.family"] = "DejaVu Sans"

# Load the data
tpot_data = pd.read_excel(r'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Models-Shap/UptakeModels-Shap/CoREMOF/CoREMOF_Uptake.xlsx', dtype=np.float64)
features = tpot_data.drop('NH3 - 1 bar (mol/kg)', axis=1)
target = tpot_data['NH3 - 1 bar (mol/kg)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=42)




# Average CV score on the training set was: -1.9948978766451961
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.4, tol=0.001)),
    RandomForestRegressor(bootstrap=False, max_features=0.1, min_samples_leaf=2, min_samples_split=3, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# Fit the TPOT pipeline
exported_pipeline.fit(X_train, y_train)

explainer = shap.Explainer(exported_pipeline.predict, X_train)
shap_values = explainer(X_test)

# Visualize the SHAP values
shap.summary_plot(shap_values, X_test, show=False)

# Get the current figure and axes objects. from @GarrettCGraham code
fig, ax = plt.gcf(), plt.gca()

ax.tick_params(labelsize=20)
ax.set_xlabel("SHAP value (impact on model output)", fontsize=20)

# Get colorbar
cb_ax = fig.axes[1] 

# Modifying color bar parameters
cb_ax.tick_params(labelsize=20)
cb_ax.set_ylabel("Feature value", fontsize=20)

plt.show()
