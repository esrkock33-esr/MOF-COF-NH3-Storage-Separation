# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:03:11 2023

@author: user
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
import matplotlib.pyplot as plt 

plt.rcParams["font.family"] = "DejaVu Sans"

# Load the data
tpot_data = pd.read_excel(r'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Models-Shap/UptakeModels-Shap/CoRECOF/CoRECOF_Uptake.xlsx', dtype=np.float64)
features = tpot_data.drop('NH3 - 1 bar (mol/kg) ', axis=1)
target = tpot_data['NH3 - 1 bar (mol/kg) ']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size=0.8, test_size=0.2, random_state=42)




# Average CV score on the training set was: -6.221492062408655
exported_pipeline = make_pipeline(
    Normalizer(norm="l1"),
    StackingEstimator(estimator=RidgeCV()),
    RandomForestRegressor(bootstrap=True, max_features=0.9000000000000001, min_samples_leaf=7, min_samples_split=5, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

# Fit the TPOT pipeline
exported_pipeline.fit(X_train, y_train)

# Extract the final model from the pipeline
final_model = exported_pipeline[-1]

# Create a callable function for SHAP analysis
def model_predict(X):
    transformed = exported_pipeline[:-1].transform(X)
    return final_model.predict(transformed)

# Create the SHAP explainer
explainer = shap.Explainer(model_predict, X_train)

# Generate SHAP values for the testing data
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
