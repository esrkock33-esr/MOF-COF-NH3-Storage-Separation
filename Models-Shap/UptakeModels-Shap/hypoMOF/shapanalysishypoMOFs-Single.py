# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:03:11 2023

@author: user
"""
import os,numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import shap
import matplotlib.pyplot as plt 

plt.rcParams["font.family"] = "DejaVu Sans"

# Load the data
tpot_data = pd.read_excel(r'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Models-Shap/UptakeModels-Shap/hypoMOF/hypoMOF_Uptake.xlsx', dtype=np.float64)
features = tpot_data.drop('NH3 - 1 bar (mol/kg)', axis=1)
target = tpot_data['NH3 - 1 bar (mol/kg)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=42)


# Average CV score on the training set was: -5.788995341298834
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=8, max_features=0.05, min_samples_leaf=12, min_samples_split=20, n_estimators=100, subsample=0.9500000000000001)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=10, n_estimators=100)),
    RandomForestRegressor(bootstrap=True, max_features=0.7000000000000001, min_samples_leaf=16, min_samples_split=5, n_estimators=100)
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
