import os,numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score #
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns



# Target column: 'NH3 - 1 bar (mol/kg)'; all other columns are used as features
tpot_data = pd.read_excel('C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Models-Shap/UptakeModels-Shap/hypoMOF/hypoMOF_Uptake.xlsx', dtype=np.float64)
features = tpot_data.drop('NH3 - 1 bar (mol/kg)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['NH3 - 1 bar (mol/kg)'], train_size=0.8, test_size=0.2, random_state=42)



# Average CV score on the training set was: -5.788995341298834
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="huber", max_depth=8, max_features=0.05, min_samples_leaf=12, min_samples_split=20, n_estimators=100, subsample=0.9500000000000001)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=1, min_samples_split=10, n_estimators=100)),
    RandomForestRegressor(bootstrap=True, max_features=0.7000000000000001, min_samples_leaf=16, min_samples_split=5, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train = exported_pipeline.predict(training_features) #
preds = exported_pipeline.predict(testing_features)

# ==============================================================================
# EXCEL EXPORT: input features + Set + simulated / ML-predicted uptake
# Row order is identical to the input file (row 1 = row 1).
# ==============================================================================
input_path = 'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Models-Shap/UptakeModels-Shap/hypoMOF/hypoMOF_Uptake.xlsx'
y_pred_all = exported_pipeline.predict(features)

df_excel = features.copy()                                   # all feature columns
df_excel['Set'] = 'Train'                                    # which split the row belongs to
df_excel.loc[testing_features.index, 'Set'] = 'Test'
df_excel['Simulated_NH3_Uptake'] = tpot_data['NH3 - 1 bar (mol/kg)'].values   # simulated (mol/kg)
df_excel['ML_Predicted_NH3_Uptake'] = y_pred_all                              # predicted (mol/kg)

output_path = os.path.join(os.path.dirname(input_path), 'hypoMOF_Uptake_Simulated_vs_Predicted.xlsx')
df_excel.to_excel(output_path, index=False)
print(f'Excel file saved: {output_path}  ({len(df_excel)} rows)')

# ==============================================================================
# OPTIONAL: UNSEEN DATA PREDICTION (uncomment to use)
# ==============================================================================

# 1. Path to the unseen-data Excel file
#unseen_path = 'path/to/unseen_data.xlsx'

# 2. Read the whole Excel file
#df_unseen_full = pd.read_excel(unseen_path)

# 3. Use all columns except the last one as features
#X_unseen = df_unseen_full.iloc[:, :-1]

# 4. Predict with the trained model
#y_pred_unseen = exported_pipeline.predict(X_unseen)

# 5. Append the predictions as a new last column
#df_unseen_full['Unseen_Result'] = y_pred_unseen

# 6. Overwrite the file with the result (change the name if you prefer)
#df_unseen_full.to_excel(unseen_path, index=False)

#print("Predictions were added as the last column and the file was saved.")

# Accuracy Metrics
r2_train = r2_score(training_target, y_pred_train)
r2_test = r2_score(testing_target, preds)
mse_train = mean_squared_error(training_target, y_pred_train)
rmse_train = math.sqrt(mse_train)
mse_test = mean_squared_error(testing_target, preds)
rmse_test = math.sqrt(mse_test)
mae_train = mean_absolute_error(training_target, y_pred_train)
mae_test = mean_absolute_error(testing_target, preds)
coef1, _ = spearmanr(training_target, y_pred_train)
coef2, _ = spearmanr(testing_target, preds)

# Print Metrics
print(f'R²_Train: {r2_train:.4f}')
print(f'R²_Test: {r2_test:.3f}')
print(f'RMSE_Train: {rmse_train:.7f}')
print(f'RMSE_Test: {rmse_test:.7f}')
print(f'MAE_Train: {mae_train:.7f}')
print(f'MAE_Test: {mae_test:.7f}')
print(f'SRCC_Train: {coef1:.3f}')
print(f'SRCC_Test: {coef2:.3f}')

# Combine Data for Scatter Plot
df_test = pd.DataFrame(
    {'Target': testing_target, 'Predictions': preds, 'Set': 'Test'})
df_train = pd.DataFrame(
    {'Target': training_target, 'Predictions': y_pred_train, 'Set': 'Training'})

# Drawing order: the set concatenated last is drawn on top
combined_df = pd.concat([df_train, df_test])

# Create JointGrid Plot
palette = {'Training': 'fuchsia', 'Test': '#2E8B57'}
g = sns.JointGrid(data=combined_df, x='Target', y='Predictions',
                  hue='Set', space=0, ratio=4, palette=palette, height=8)

# Scatterplot
g.plot_joint(sns.scatterplot, alpha=1, edgecolor='black', s=150)

# Histogram style: layered bars (Test drawn first, Training drawn last — to be on top)
test_data = combined_df[combined_df['Set'] == 'Test']
train_data = combined_df[combined_df['Set'] == 'Training']

# Top histograms: TRAINING (drawn last, appears in front)
sns.histplot(data=train_data, x='Target', ax=g.ax_marg_x,
             bins=50, color='fuchsia', multiple='layer')
sns.histplot(data=train_data, y='Predictions', ax=g.ax_marg_y,
             bins=50, color='fuchsia', multiple='layer')

# Bottom histograms: TEST
sns.histplot(data=test_data, x='Target', ax=g.ax_marg_x,
             bins=50, color='#2E8B57', multiple='layer')
sns.histplot(data=test_data, y='Predictions', ax=g.ax_marg_y,
             bins=50, color='#2E8B57', multiple='layer')

# Automatic axis limits from the data range (# Automatic axis limits: start at 0, 5% padding above the maximum)
data_min = min(combined_df['Target'].min(), combined_df['Predictions'].min())
data_max = max(combined_df['Target'].max(), combined_df['Predictions'].max())
pad = 0.05 * data_max
axis_min, axis_max = 0, data_max + pad   # axes start exactly at (0, 0)

# Round tick values (about 5 ticks)
ticks = MaxNLocator(nbins=5).tick_values(axis_min, axis_max)
ticks = ticks[(ticks >= axis_min) & (ticks <= axis_max)]

g.ax_joint.set_xlim(axis_min, axis_max)
g.ax_joint.set_ylim(axis_min, axis_max)
g.ax_joint.set_xticks(ticks)
g.ax_joint.set_yticks(ticks)
g.ax_joint.set_aspect('equal', adjustable='box')
g.ax_joint.tick_params(axis='both', labelsize=30, length=6, width=1.5)

# Format plot spines
for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)

# Increase marker size
g.ax_joint.collections[0].set_sizes([150])

# Set Axis Labels
g.set_axis_labels(
    r'Simulated $\mathrm{NH_3}$ Uptake ($\mathrm{mol/kg}$)',
    r'ML-Predicted $\mathrm{NH_3}$ Uptake ($\mathrm{mol/kg}$)',
    fontsize=20
)


# Diagonal Reference Line
g.ax_joint.plot([axis_min, axis_max], [axis_min, axis_max],
                color='black', linestyle='--')

# Remove Legend
g.ax_joint.legend_.remove()


# Number format: 3 decimals for small values, 1 decimal for large ones
def fmt(v):
    return f'{v:.3f}' if abs(v) < 100 else f'{v:.1f}'

# Training metrics (fuchsia, upper left)
g.ax_joint.text(0.02, 0.94, r'$\bf{Train}$', fontsize=32,
                color='fuchsia', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.02, 0.87, f'R²: {r2_train:.3f}', fontsize=32,
                color='fuchsia', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.02, 0.80, f'RMSE: {fmt(rmse_train)}', fontsize=32,
                color='fuchsia', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.02, 0.73, f'MAE: {fmt(mae_train)}', fontsize=32,
                color='fuchsia', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.02, 0.66, f'SRCC: {coef1:.3f}', fontsize=32,
                color='fuchsia', transform=g.ax_joint.transAxes)

# Test metrics (green, lower right)
g.ax_joint.text(0.49, 0.29, r'$\bf{Test}$', fontsize=32,
                color='#2E8B57', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.49, 0.22, f'R²: {r2_test:.3f}', fontsize=32,
                color='#2E8B57', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.49, 0.15, f'RMSE: {fmt(rmse_test)}', fontsize=32,
                color='#2E8B57', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.49, 0.08, f'MAE: {fmt(mae_test)}', fontsize=32,
                color='#2E8B57', transform=g.ax_joint.transAxes)
g.ax_joint.text(0.49, 0.01, f'SRCC: {coef2:.3f}', fontsize=32,
                color='#2E8B57', transform=g.ax_joint.transAxes)


# Show Plot
plt.show()



