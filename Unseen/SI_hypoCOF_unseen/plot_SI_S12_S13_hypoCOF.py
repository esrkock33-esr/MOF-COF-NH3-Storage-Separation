# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:54:20 2026

@author: nemo26
"""

# -*- coding: utf-8 -*-
"""
Figure S12 (NH3 uptake) and Figure S13 (APS = 10**logAPS, log y-axis): simulated (gray) vs ML-predicted unseen (green) hypoCOFs
against PLD and PV.  Reads ONLY Excel files: the simulated training sets and the *_prediction_NEW.xlsx files
written by hypoCOF_remaining_predict.py (no npz).
Output: PNG (600 dpi) + PDF + uncompressed RGB TIFF.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from PIL import Image

FOLDER = 'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Unseen/SI_hypoCOF_unseen'
SIM_SINGLE  = os.path.join(FOLDER, 'HypoCOF_Uptake.xlsx')                          # simulated, target 'NH3 - 1 bar (mol/kg)'
SIM_MIX     = os.path.join(FOLDER, 'HypoCOF_Mixture.xlsx')                         # simulated, target 'logAPS'
PRED_SINGLE = os.path.join(FOLDER, 'Remaining_48496_Single_prediction_NEW.xlsx')   # column 'ML_Predicted_NH3_Uptake'
PRED_MIX    = os.path.join(FOLDER, 'Remaining_48496_Mixture_prediction_NEW.xlsx')  # column 'ML_Predicted_logAPS'
PLD, PV = 'PLD (Å)', 'Pore Volume (cm3/g)'
N_STARS = 10                        # stars = the N highest structures of the COMBINED set (simulated + ML-predicted); 0: no stars
FONT = os.environ.get('TEST_FONT', 'Times New Roman')
GRAY, GREEN = '#7f7f7f', '#2ecc71'  # simulated / hypoCOF colour (same as the other figures)
FIG_W, FIG_H = 16.5 / 2.54, 7.4 / 2.54

if FONT not in {f.name for f in font_manager.fontManager.ttflist}:
    sys.exit(f'Font "{FONT}" not found - figure NOT drawn.')
plt.rcParams.update({'font.family': FONT, 'font.size': 8, 'axes.linewidth': 0.8, 'mathtext.fontset': 'custom',
                     'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic', 'mathtext.bf': f'{FONT}:bold',
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'pdf.fonttype': 42})

def read_clean(p):
    d = pd.read_excel(p); d.columns = d.columns.astype(str).str.strip(); return d

s_sim, m_sim = read_clean(SIM_SINGLE), read_clean(SIM_MIX)
s_pred, m_pred = read_clean(PRED_SINGLE), read_clean(PRED_MIX)
n_sim, n_rem = len(s_sim), len(s_pred)
assert len(m_sim) == n_sim and len(m_pred) == n_rem, 'single and mixture files have different numbers of structures'
print(f'simulated n = {n_sim} | ML-predicted unseen n = {n_rem}')

def make(tag, sim, ysim, pred, yrem, ylabel, name, ylog=False):
    ysim, yrem = np.asarray(ysim, float), np.asarray(yrem, float)
    yr_all = yrem
    # combined ranking: which of the N highest structures are simulated and which are ML-predicted?
    yy = np.r_[ysim, yr_all]; src = np.r_[np.zeros(len(ysim), int), np.ones(len(yr_all), int)]
    top = np.argsort(yy)[::-1][:N_STARS]
    n_top_sim, n_top_pred = int((src[top] == 0).sum()), int((src[top] == 1).sum())
    print(f'{tag}: among the {N_STARS} highest structures: {n_top_sim} simulated, {n_top_pred} ML-predicted')
    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    for ax, (xs, xr, xl, let) in zip(axs, ((sim[PLD].values, pred[PLD].values, 'PLD (Å)', '(a)'),
                                          (sim[PV].values, pred[PV].values, r'PV (cm$^{3}$/g)', '(b)'))):
        xx = np.r_[xs, xr]
        order = np.random.RandomState(42).permutation(len(xx))          # random draw order: both groups stay visible
        ax.scatter(xx[order], yy[order], s=1.2, c=np.where(src[order] == 0, GRAY, GREEN), alpha=0.5, linewidths=0, rasterized=True, zorder=2)
        if N_STARS:
            ax.scatter(xx[top], yy[top], marker='*', s=55, c=np.where(src[top] == 0, GRAY, GREEN), edgecolors='black', linewidths=0.6, zorder=5)
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylabel(ylabel, fontsize=8.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        if ylog: ax.set_yscale('log')            # APS spans several orders of magnitude
        else: ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.minorticks_off()                       # no small (minor) ticks on the log y-axis
        ax.tick_params(labelsize=8); ax.set_xlim(left=0)
        ax.text(-0.02, 1.04, let, transform=ax.transAxes, fontweight='bold', fontsize=10, ha='right', va='bottom')
    h = [Line2D([], [], marker='o', ls='', color=GRAY, ms=5, label=f'Simulated (n = {n_sim})'),
         Line2D([], [], marker='o', ls='', color=GREEN, ms=5, label=f'ML-predicted, unseen (n = {n_rem})')]
    if N_STARS:
        if n_top_sim:  h.append(Line2D([], [], marker='*', ls='', mfc=GRAY, mec='black', mew=0.6, ms=8, label=f'Top {N_STARS}: {n_top_sim} simulated'))
        if n_top_pred: h.append(Line2D([], [], marker='*', ls='', mfc=GREEN, mec='black', mew=0.6, ms=8, label=f'Top {N_STARS}: {n_top_pred} ML-predicted'))
    axs[1].legend(handles=h, loc='upper right', frameon=False, fontsize=7.5, handletextpad=0.3, borderpad=0.2)
    fig.tight_layout(w_pad=1.5)
    os.makedirs(FOLDER, exist_ok=True)
    base = os.path.join(FOLDER, name)
    fig.savefig(base + '.png', dpi=600, facecolor='white'); fig.savefig(base + '.pdf', facecolor='white')
    with Image.open(base + '.png') as im: im.convert('RGB').save(base + '.tif', dpi=(600, 600))
    plt.close(fig)
    print('Saved:', base + '.(png/pdf/tif)')

make('S12', s_sim, s_sim['NH3 - 1 bar (mol/kg)'], s_pred, s_pred['ML_Predicted_NH3_Uptake'],
     r'NH$_3$ uptake (mol/kg)', 'Figure_S12_hypoCOF_uptake_PLD_PV')
make('S13', m_sim, 10.0 ** m_sim['logAPS'], m_pred, 10.0 ** m_pred['ML_Predicted_logAPS'],
     'APS (mol/kg)', 'Figure_S13_hypoCOF_APS_PLD_PV', ylog=True)