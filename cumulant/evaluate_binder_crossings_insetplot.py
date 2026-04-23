import glob, numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as sp
import seaborn as sns
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("poster")
import pickle
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import LASSI_2 as lassi
import BinTrjProc as BR
import TrjProc as TP
import argparse

# Set global plotting parameters
plt.rcParams.update({
    'font.family': "sans-serif",
    #'font.family':"DejaVu Sans",
    'font.size': 14,
    'axes.labelsize': 22,
    'axes.titlesize': 22,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'legend.frameon': False,
    'legend.framealpha': 0.8,
    'axes.linewidth': 2,
    'lines.linewidth': 2,
    #'xtick.direction': 'in',
    #'ytick.direction': 'in',
    #'xtick.top': True,
    #'ytick.right': True,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
})

from scipy.stats import kurtosis
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar, brentq
from itertools import combinations

Nchains=10000
variant='WTARO'
cmap = plt.cm.plasma_r
Nsubboxes=10000

def binder_about_mean(x, mean0=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    mu = float(mean0)
    y  = x - mu
    mu2 = np.mean(y*y)
    mu4 = np.mean(y*y*y*y)
    return 1.0 - mu4/(3.0*mu2*mu2)

def jackknife_binder_over_frames(frames_2d, mean0=None, nblocks=None):
    X = np.asarray(frames_2d, float)
    n_frames = X.shape[0]
    U_full = binder_about_mean(X.ravel(), mean0)
    if n_frames < 2:
        return U_full, np.nan
    blocks = np.array_split(np.arange(n_frames), nblocks)
    U_jk = np.empty(len(blocks))
    for i, drop in enumerate(blocks):
        keep = np.ones(n_frames, dtype=bool); keep[drop] = False
        U_jk[i] = binder_about_mean(X[keep].ravel(), mean0)
    U_mean = U_jk.mean()
    err = np.sqrt((len(blocks)-1) * np.mean((U_jk - U_mean)**2))
    return U_full, err



L_boxes = [size for size in range(160, 191, 10)]
T_strings = ["58.75","59","59.25","59.50","59.75"]

Nchains   = 10000
Lsim      = 240
Nsubboxes = 10000
variant   = variant 
mean_rho  = float((Nchains * 137) / (Lsim**3))

curves = {}
for L_box in L_boxes:
    U_vals, U_errs, T_kept = [], [], []
    for t_str in T_strings:
        pattern = (f'Density_subbox_Nchains{Nchains}_6e11_last100snaps/{variant}/Lbox{Lsim}/'
                   f'Density_T{t_str}_L{L_box}_frame*_Nsubboxes{Nsubboxes}.npy')
        files = sorted(glob.glob(pattern))
        if not files:
            continue
        frames = [np.load(fp, allow_pickle=False) for fp in files]
        frames = np.vstack(frames)
        U4, U4_err = jackknife_binder_over_frames(frames, mean0=mean_rho, nblocks=min(100, len(frames)))
        U_vals.append(U4)
        U_errs.append(U4_err)
        T_kept.append(float(t_str))

    if T_kept:
        order = np.argsort(T_kept)
        curves[L_box] = {
            "T": np.array(T_kept)[order],
            "U": np.array(U_vals)[order],
            "E": np.array(U_errs)[order],
        }
        print(f"L={L_box}: {len(T_kept)} temps")

interps = {}
for L_box, d in curves.items():
    ok = np.isfinite(d["T"]) & np.isfinite(d["U"])
    T_i, U_i = d["T"][ok], d["U"][ok]
    if T_i.size >= 4:
        interps[L_box] = PchipInterpolator(T_i, U_i, extrapolate=False)

if len(interps) < 2:
    raise RuntimeError("Need ≥2 sizes with valid data for crossings.")

T_lo = max(d["T"].min() for d in curves.values())
T_hi = min(d["T"].max() for d in curves.values())
T_grid = np.linspace(T_lo, T_hi, 5000)

def crossing_T(f1, f2, Tmin, Tmax, U_window=(0.2, 0.6), T_ref=None):
    def diff(T): return f1(T) - f2(T)
    T_vals = np.linspace(Tmin, Tmax, 5000)
    diffs = diff(T_vals)
    sign_changes = np.where(np.sign(diffs[:-1]) != np.sign(diffs[1:]))[0]
    roots = []
    for idx in sign_changes:
        a, b = T_vals[idx], T_vals[idx+1]
        try:
            Tc = brentq(diff, a, b)
            U_mid = 0.5 * (f1(Tc) + f2(Tc))
            if U_window[0] <= U_mid <= U_window[1]:
                roots.append(Tc)
        except ValueError:
            pass

    if T_ref is not None and len(roots) > 1:
        roots = [min(roots, key=lambda x: abs(x - T_ref))]
    return roots

Tc_pairs, pair_labels = [], []
L_sorted = sorted(interps.keys())
for (L1, L2) in combinations(L_sorted, 2):
    f1, f2 = interps[L1], interps[L2]
    Tmin = max(curves[L1]['T'].min(), curves[L2]['T'].min())
    Tmax = min(curves[L1]['T'].max(), curves[L2]['T'].max())
    roots = crossing_T(f1, f2, Tmin, Tmax)
    for r in roots:
        if T_lo <= r <= T_hi:
            Tc_pairs.append(r)
            pair_labels.append(f"{L1}-{L2}")

Tc_pairs = np.array(Tc_pairs)
Tc_mean = np.mean(Tc_pairs)
Tc_std  = np.std(Tc_pairs)
Tc_min, Tc_max = np.min(Tc_pairs), np.max(Tc_pairs)

print(f"\nPairwise crossings found: {len(Tc_pairs)}")
for lbl, Tc_ij in zip(pair_labels, Tc_pairs):
    print(f"  {lbl} → T_c = {Tc_ij:.3f}")
print(f"\nT_c (mean ± std): {Tc_mean:.3f} ± {Tc_std:.3f}")
print(f"T_c range: [{Tc_min:.3f}, {Tc_max:.3f}]")

Tc_median = np.median(sorted(Tc_pairs))
print(f"T_c (median): {Tc_median}")

cmap = plt.cm.plasma_r
norm = mcolors.Normalize(vmin=0, vmax=len(L_boxes)-1)

plt.figure(figsize=(7.5,4.5), dpi=300)

for k, L_box in enumerate(sorted(interps.keys())):
    d = curves[L_box]
    color = cmap(norm(k))
    plt.errorbar(np.array(d["T"])*5.6, d["U"], yerr=d["E"], marker='o', ms=8, capsize=10,
                 color=color, lw=1.8, elinewidth=1.3, label=r'$\mathrm{L = }$'+f"{L_box}")

plt.ylabel(r'$U_{L}(T) = 1 - \frac{\langle \Delta \phi_L^4 \rangle}{3\langle \Delta \phi_L^2 \rangle^2}$')
plt.xlabel(r'$T\;(\mathrm{K})$')
plt.xticks(np.array([58.6,58.8,59,59.2,59.4,59.6,59.8])*5.6)
plt.axvline(59.36*5.6,ls='--',color='k',linewidth=2.0)
plt.axhline(0.466,ls=':',color='k',lw=2.0)
plt.scatter(59.36*5.6,0.466,s=400,marker='x',linewidths=4,color='green',zorder=10)
plt.text(58.75*5.6,0.40,r'$U^\ast\!\approx\!0.466$',color='green',fontsize=35)
plt.axvspan(59.170*5.6, 59.399*5.6,color='lightgray', alpha=0.5)
plt.tight_layout()
plt.savefig(f'bindercumulant_pairwisecrossing_insetzoomedplot.pdf', bbox_inches='tight')

