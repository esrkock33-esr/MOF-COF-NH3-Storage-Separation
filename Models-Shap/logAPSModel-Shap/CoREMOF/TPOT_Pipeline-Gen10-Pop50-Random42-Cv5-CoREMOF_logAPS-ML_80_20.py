import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score #
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
input_path = r'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Models-Shap/logAPSModel-Shap/CoREMOF/CoREMOF_Mixture_logAPS.xlsx'
tpot_data = pd.read_excel(input_path, dtype=np.float64)
features = tpot_data.drop('logAPS', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['logAPS'],train_size=0.8, test_size=0.2, random_state=42)



# Average CV score on the training set was: -0.022128527060063224
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    MinMaxScaler(),
    XGBRegressor(learning_rate=0.1, max_depth=8, min_child_weight=16, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=1.0, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train = exported_pipeline.predict(training_features) #
preds = exported_pipeline.predict(testing_features)

# =====================================================================
# EXCEL EXPORT: Input data (all features) + Set + logAPS (sim/pred) + APS (sim/pred)
# Row order is identical to the input file (row 1 = row 1).
# Both plots below are drawn directly from this table.
# =====================================================================
y_pred_all = exported_pipeline.predict(features)

df_excel = features.copy()                                          # all feature columns
df_excel['Set'] = 'Train'                                           # which split the row belongs to
df_excel.loc[testing_features.index, 'Set'] = 'Test'
df_excel['Simulated_logAPS'] = tpot_data['logAPS'].values           # original target (log scale)
df_excel['ML_Predicted_logAPS'] = y_pred_all                         # model output (log scale)
output_path = os.path.join(os.path.dirname(input_path), 'CoREMOFMix_logAPS_Simulated_vs_Predicted.xlsx')
df_excel.to_excel(output_path, index=False)
print(f'Excel file saved: {output_path}  ({len(df_excel)} rows)')
# =====================================================================
# =====================================================================
# PARITY PLOT FUNCTION
# Same metrics and same plot style as before; it is only wrapped in a
# function so it can be drawn twice (logAPS and APS) from the Excel table.
# =====================================================================
def parity_plot(sim_col, pred_col, x_label, y_label, log_axes=False, test_text_x=0.49, title=''):
    train_df = df_excel[df_excel['Set'] == 'Train']
    test_df = df_excel[df_excel['Set'] == 'Test']
    training_target, y_pred_train = train_df[sim_col], train_df[pred_col].values
    testing_target, preds = test_df[sim_col], test_df[pred_col].values

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
    print(f'\n===== {title} =====')
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

    # NOKTALARIN ÖNE VEYA ARKAYA GİTMESİ BURADAN AYARLANIYOR!!
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
                 bins=50, color='fuchsia', multiple='layer', log_scale=log_axes)
    sns.histplot(data=train_data, y='Predictions', ax=g.ax_marg_y,
                 bins=50, color='fuchsia', multiple='layer', log_scale=log_axes)

    # Bottom histograms: TEST
    sns.histplot(data=test_data, x='Target', ax=g.ax_marg_x,
                 bins=50, color='#2E8B57', multiple='layer', log_scale=log_axes)
    sns.histplot(data=test_data, y='Predictions', ax=g.ax_marg_y,
                 bins=50, color='#2E8B57', multiple='layer', log_scale=log_axes)

    data_min = min(combined_df['Target'].min(), combined_df['Predictions'].min())
    data_max = max(combined_df['Target'].max(), combined_df['Predictions'].max())

    if log_axes:
        # Log-scale axes: limits rounded to full powers of 10, ticks at 10^n
        exp_min = math.floor(np.log10(data_min))
        exp_max = math.ceil(np.log10(data_max))
        axis_min, axis_max = 10.0 ** exp_min, 10.0 ** exp_max
        step = max(1, math.ceil((exp_max - exp_min) / 5))   # at most ~6 labels so they don't overlap
        ticks = 10.0 ** np.arange(exp_min, exp_max + 1, step)
        g.ax_joint.set_xscale('log')
        g.ax_joint.set_yscale('log')
    else:
        # Automatic axis limits based on the data range (5% padding on each side)
        pad = 0.05 * (data_max - data_min)
        axis_min, axis_max = data_min - pad, data_max + pad
        # Nice round tick values (about 5 ticks)
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
    g.set_axis_labels(x_label, y_label, fontsize=28)

    # Diagonal Reference Line
    g.ax_joint.plot([axis_min, axis_max], [axis_min, axis_max],
                    color='black', linestyle='--')

    # Remove Legend
    g.ax_joint.legend_.remove()

    # Number format for the plot text: 3 decimals for small values, 1 decimal for large ones
    def fmt(v):
        return f'{v:.3f}' if abs(v) < 100 else f'{v:.1f}'

    # "Train" as bold, fuchsia
    g.ax_joint.text(0.02, 0.94, r'$\bf{Train}$', fontsize=32,
                    color='fuchsia', transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.02, 0.87, f'R²: {r2_train:.3f}', fontsize=32,
                    color='fuchsia', transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.02, 0.80, f'RMSE: {fmt(rmse_train)}',
                    fontsize=32, color='fuchsia', transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.02, 0.73, f'MAE: {fmt(mae_train)}', fontsize=32,
                    color='fuchsia', transform=g.ax_joint.transAxes)
    g.ax_joint.text(0.02, 0.66, f'SRCC: {coef1:.3f}', fontsize=32,
                    color='fuchsia', transform=g.ax_joint.transAxes)

    # "Test" as bold, green (block moved up one line to make room for SRCC)
    g.ax_joint.text(test_text_x, 0.29, r'$\bf{Test}$', fontsize=32,
                    color='#2E8B57', transform=g.ax_joint.transAxes)
    g.ax_joint.text(test_text_x, 0.22, f'R²: {r2_test:.3f}', fontsize=32,
                    color='#2E8B57', transform=g.ax_joint.transAxes)
    g.ax_joint.text(test_text_x, 0.15, f'RMSE: {fmt(rmse_test)}', fontsize=32,
                    color='#2E8B57', transform=g.ax_joint.transAxes)
    g.ax_joint.text(test_text_x, 0.08, f'MAE: {fmt(mae_test)}', fontsize=32,
                    color='#2E8B57', transform=g.ax_joint.transAxes)
    g.ax_joint.text(test_text_x, 0.01, f'SRCC: {coef2:.3f}', fontsize=32,
                    color='#2E8B57', transform=g.ax_joint.transAxes)

    # Show Plot
    plt.show()


# =====================================================================
# PLOT 1: logAPS (the scale the model was trained on) — linear axes
# =====================================================================
parity_plot('Simulated_logAPS', 'ML_Predicted_logAPS',
            r'Simulated logAPS', r'ML-Predicted logAPS',
            log_axes=False, test_text_x=0.49, title='logAPS scale')
