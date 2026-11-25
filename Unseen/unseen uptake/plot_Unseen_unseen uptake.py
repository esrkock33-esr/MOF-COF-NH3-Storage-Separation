# -*- coding: utf-8 -*-
"""
Unseen-data predictions (NH3 uptake, 1 bar): one separate figure per unseen dataset.
Same look as the Figure S10 panels: parity plot + marginal histograms, dataset colour,
metric block (R2, MAE, SRCC) and "model database (n) -> unseen database (n)" title.
Reads ONLY the unseen Excel files written by the TPOT scripts (no npz).
Each figure is saved next to its Excel as PNG (600 dpi) + PDF + TIFF.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from PIL import Image

BASE = 'C:/Users/nemo26/Documents/GitHub/MOF-COF-NH3-Storage-Separation/Unseen/unseen uptake'
PANELS = [  # model database, UNSEEN database, unseen Excel, colour of the unseen database, output name
    ('CoRE MOF', 'CoRE COF', f'{BASE}/(a) CoREMOF-to-CoRECOF/CoREMOF_to_CoRECOF_unseen_Simulated_vs_Predicted.xlsx', '#3498db', 'Unseen_a_CoREMOF_to_CoRECOF'),
    ('hypoMOF',  'hypoCOF',  f'{BASE}/(b)hypoMOF-to-hypoCOF/hypoMOF_to_hypoCOF_unseen_Simulated_vs_Predicted.xlsx',   '#2ecc71', 'Unseen_b_hypoMOF_to_hypoCOF'),
    ('CoRE MOF', 'hypoMOF',  f'{BASE}/(c) CoRE MOF-to-hypoMOF/CoREMOF_to_hypoMOF_unseen_Simulated_vs_Predicted.xlsx', '#9b59b6', 'Unseen_c_CoREMOF_to_hypoMOF'),
    ('CoRE COF', 'hypoCOF',  f'{BASE}/(d) CoRE COF-to-hypoCOF/CoRECOF_to_hypoCOF_unseen_Simulated_vs_Predicted.xlsx', '#2ecc71', 'Unseen_d_CoRECOF_to_hypoCOF'),
]   # a missing Excel is skipped (check the file names of b, c, d)
MODEL_N = {'CoRE MOF': 1860, 'hypoMOF': 5253, 'CoRE COF': 1086, 'hypoCOF': 19886}   # structures in the model (training) database

FONT = 'Times New Roman'
FS_BASE, FS_LABEL, FS_TITLE, FS_METRIC = 8, 8.5, 9.5, 8
DECIMALS = 3
FIG_W = 16.5 / 2 / 2.54             # one S10 column: 8.25 cm wide
FONT = os.environ.get('TEST_FONT', FONT)
if FONT not in {f.name for f in font_manager.fontManager.ttflist}:
    sys.exit(f'Font "{FONT}" not found - figure NOT drawn.')
plt.rcParams.update({'font.family': FONT, 'font.size': FS_BASE, 'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.labelsize': FS_BASE, 'ytick.labelsize': FS_BASE, 'mathtext.fontset': 'custom',
    'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic', 'mathtext.bf': f'{FONT}:bold', 'mathtext.sf': FONT,
    'mathtext.cal': FONT, 'mathtext.tt': FONT, 'mathtext.fallback': None, 'pdf.fonttype': 42})
X_LABEL = r'Sim-$\mathbf{N_{NH_3}}$ (mol/kg)'
Y_LABEL = r'ML-$\mathbf{N_{NH_3}}$ (mol/kg)'

# layout (inches), same as one S10 panel
LEFT, YLAB, M, RGT = 0.06, 0.46, 0.15, 0.10
J = FIG_W - LEFT - YLAB - M - RGT    # parity square side
TOPB, XBAND = 0.34, 0.46
H = TOPB + M + J + XBAND

def darken(c, f=0.72):
    r, g, b = mcolors.to_rgb(c); return (r * f, g * f, b * f)

def load_excel(path):
    d = pd.read_excel(path); d.columns = d.columns.astype(str).str.strip()
    sim  = [c for c in ('Simulated_NH3_Uptake_1bar', 'Simulated_NH3_Uptake') if c in d.columns]
    pred = [c for c in ('Unseen_Result', 'ML_Predicted_NH3_Uptake') if c in d.columns]
    if not sim or not pred:
        raise KeyError(f'{path}: simulated / predicted column not found. Columns: {list(d.columns)}')
    return d[sim[0]].astype(float).values, d[pred[0]].astype(float).values

for mod, uns, path, col, name in PANELS:
    if not os.path.exists(path):
        print('[skipped, file not found]', path); continue
    y, p = load_excel(path)
    n = len(y)
    r2, mae, srcc = r2_score(y, p), mean_absolute_error(y, p), spearmanr(y, p)[0]
    print(f'{mod} -> {uns}: n = {n} | R2 {r2:.3f} | MAE {mae:.3f} | SRCC {srcc:.3f}')

    fig = plt.figure(figsize=(FIG_W, H))
    def axes_in(x, yy, w, h): return fig.add_axes([x / FIG_W, 1 - (yy + h) / H, w / FIG_W, h / H])
    x0 = LEFT + YLAB
    ax  = axes_in(x0, TOPB + M, J, J)
    axx = axes_in(x0, TOPB, J, M)
    axy = axes_in(x0 + J, TOPB + M, M, J)
    for a in (axx, axy): a.axis('off')

    vmax = 1.05 * max(y.max(), p.max())
    s, lw = (10, 0.25) if n < 2000 else ((5, 0.15) if n < 6000 else (3, 0.1))
    ax.plot([0, vmax], [0, vmax], ls='--', lw=0.7, color='black', zorder=1)
    ax.scatter(y, p, s=s, color=col, edgecolors='black', linewidths=lw, zorder=2, rasterized=True)
    t = MaxNLocator(nbins=4).tick_values(0, vmax); t = t[(t >= 0) & (t <= vmax)]
    ax.set_xlim(0, vmax); ax.set_ylim(0, vmax); ax.set_xticks(t); ax.set_yticks(t)
    ax.set_xlabel(X_LABEL, fontsize=FS_LABEL, fontweight='bold', labelpad=3)
    ax.set_ylabel(Y_LABEL, fontsize=FS_LABEL, fontweight='bold', labelpad=3)
    bins = np.linspace(0, vmax, 41)
    axx.hist(y, bins=bins, color=col, edgecolor='black', linewidth=0.2); axx.set_xlim(0, vmax)
    axy.hist(p, bins=bins, color=col, edgecolor='black', linewidth=0.2, orientation='horizontal'); axy.set_ylim(0, vmax)

    # metric block (upper-left, white backing)
    f = f'.{DECIMALS}f'
    for i, (lab, v) in enumerate((('R²', r2), ('MAE', mae), ('SRCC', srcc))):
        yy = 0.945 - i * 0.075
        ax.text(0.04, yy, lab, transform=ax.transAxes, fontsize=FS_METRIC, fontweight='bold', va='center', zorder=5)
        ax.text(0.30, yy, f'{v:{f}}'.replace('-', '−'), transform=ax.transAxes, fontsize=FS_METRIC, fontweight='bold',
                color=darken(col), va='center', zorder=5)
    ax.add_patch(plt.Rectangle((0.015, 0.755), 0.45, 0.225, transform=ax.transAxes, fc='white', ec='none', alpha=0.85, zorder=4))

    # title: model database (n) -> unseen database (n), centred on the parity axis
    ty = 1 - (TOPB / 2 + 0.02) / H
    gray, dk = '#4d4d4d', darken(col)
    pieces = [(mod, FS_TITLE, 'bold', '#222222'), (f'(n = {MODEL_N[mod]})', FS_BASE, 'normal', gray), ('→', FS_TITLE, 'bold', '#222222'),
              (uns, FS_TITLE, 'bold', dk), (f'(n = {n})', FS_BASE, 'normal', gray)]
    arts = [fig.text(0, ty, tx, fontsize=fs, fontweight=fw, color=cl, ha='left', va='center') for tx, fs, fw, cl in pieces]
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    gap = 0.045 * fig.dpi
    w = [a.get_window_extent(rend).width for a in arts]
    x = (x0 + J / 2) * fig.dpi - (sum(w) + gap * (len(arts) - 1)) / 2
    for a, wi in zip(arts, w):
        a.set_x(x / fig.bbox.width); x += wi + gap

    base = os.path.join(os.path.dirname(path), name)
    fig.savefig(base + '.png', dpi=600, facecolor='white'); fig.savefig(base + '.pdf', facecolor='white')
    with Image.open(base + '.png') as im: im.convert('RGB').save(base + '.tif', dpi=(600, 600))
    plt.close(fig)
    print('Saved:', base + '.(png/pdf/tif)')
