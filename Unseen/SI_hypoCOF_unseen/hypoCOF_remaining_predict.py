# -*- coding: utf-8 -*-
"""
hypoCOF: predict the REMAINING (unsimulated) 48,496 hypoCOFs with the CURRENT single-component (NH3 uptake)
and mixture (log APS) models, and save the predictions as Excel files (no npz).

TPOT is NOT run: the two exported pipelines are only re-fitted on the 80 % training split (random_state=42)
of the current hypoCOF data and then used to predict the remaining structures.

Put the 4 Excel files below into FOLDER (or edit the names), run this script ONCE, then run
plot_SI_S12_S13_hypoCOF.py (it reads the training Excels and the *_prediction_NEW.xlsx files written here).
"""
import os, math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from tpot.builtins import OneHotEncoder, StackingEstimator
from tpot.export_utils import set_param_recursive
from xgboost import XGBRegressor

FOLDER = 'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Unseen/SI_hypoCOF_unseen'
TRAIN_SINGLE = os.path.join(FOLDER, 'HypoCOF_Uptake.xlsx')                   # 19,886 simulated, target: NH3 - 1 bar (mol/kg)
TRAIN_MIX    = os.path.join(FOLDER, 'HypoCOF_Mixture.xlsx')                    # 19,886 simulated, target: logAPS
REM_SINGLE   = os.path.join(FOLDER, 'Remaining_48496_Single_features.xlsx')      # 48,496 remaining (Structure + features)
REM_MIX      = os.path.join(FOLDER, 'Remaining_48496_Mixture_features.xlsx')

def read_clean(p):
    d = pd.read_excel(p); d.columns = d.columns.astype(str).str.strip(); return d

def fit_predict(train_xlsx, target, rem_xlsx, pipeline, pred_cols_to_drop):
    d = read_clean(train_xlsx).astype(np.float64)
    X, y = d.drop(target, axis=1), d[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
    set_param_recursive(pipeline.steps, 'random_state', 42) if hasattr(pipeline, 'steps') else setattr(pipeline, 'random_state', 42)
    pipeline.fit(Xtr, ytr)
    for nm, a, b in (('Train', Xtr, ytr), ('Test', Xte, yte)):
        p = pipeline.predict(a)
        print(f'  {nm:5s} n={len(b):6d}  R2 {r2_score(b, p):.3f}  MAE {mean_absolute_error(b, p):.3f}  SRCC {spearmanr(b, p)[0]:.3f}')
    rem = read_clean(rem_xlsx)
    missing = [c for c in X.columns if c not in rem.columns]
    assert not missing, f'remaining file lacks columns: {missing}'
    Xr = rem[list(X.columns)].astype(np.float64)             # same names, same order as in training
    assert not Xr.isna().any().any()
    return d, X, y, rem, Xr, pipeline.predict(Xr), pipeline.predict(X)

# ------------------------------------------------------------------ single-component (NH3 uptake, 1 bar)
print('Single-component model (NH3 uptake)')
pipe_s = XGBRegressor(learning_rate=0.1, max_depth=9, min_child_weight=5, n_estimators=100, n_jobs=1,
                      objective='reg:squarederror', subsample=0.6500000000000001, verbosity=0)
ds, Xs, ys, rem_s, Xrs, pred_s, _ = fit_predict(TRAIN_SINGLE, 'NH3 - 1 bar (mol/kg)', REM_SINGLE, pipe_s, None)

# ------------------------------------------------------------------ mixture (log APS)
print('Mixture model (log APS)')
pipe_m = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    OneHotEncoder(minimum_fraction=0.05, sparse=False, threshold=10),
    StackingEstimator(estimator=RidgeCV()),
    XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=13, n_estimators=100, n_jobs=1,
                 objective='reg:squarederror', subsample=0.7500000000000001, verbosity=0))
dm, Xm, ym, rem_m, Xrm, pred_m, _ = fit_predict(TRAIN_MIX, 'logAPS', REM_MIX, pipe_m, None)

# ------------------------------------------------------------------ checks + save
print('\nStructures: simulated', len(ds), '| remaining', len(rem_s))
assert len(rem_s) == len(rem_m) and (rem_s['Structure'].values == rem_m['Structure'].values).all(), 'single/mixture remaining lists differ'

for src, nm, col, pr in ((rem_s, 'Single', 'ML_Predicted_NH3_Uptake', pred_s), (rem_m, 'Mixture', 'ML_Predicted_logAPS', pred_m)):
    out = src.drop(columns=[c for c in src.columns if c.startswith('ML_Predicted') or c == 'LCD_placeholder']).copy()
    out[col] = pr
    if nm == 'Mixture': out['ML_Predicted_APS'] = 10.0 ** pr
    p = os.path.join(FOLDER, f'Remaining_48496_{nm}_prediction_NEW.xlsx'); out.to_excel(p, index=False); print('Saved:', p)