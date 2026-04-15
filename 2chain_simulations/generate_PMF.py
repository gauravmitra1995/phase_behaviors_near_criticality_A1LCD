import os
import sys
import argparse
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from skimage import measure
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("poster")
import pickle
import time
import sys
import LASSI_2 as lassi
import BinTrjProc as BR
import TrjProc as TP
from scipy.signal import savgol_filter
from pathlib import Path

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


def flatten_2d_list(two_d_list):
    flattened_list = [item for sublist in two_d_list for item in sublist]
    return flattened_list


def process_pmf(filename, temp, n_bins, bin_cut, r_cut):
    Free_energy = np.loadtxt(filename, skiprows=1)

    nb = min(n_bins, Free_energy.shape[0])
    r = Free_energy[:nb, 0]
    F = Free_energy[:nb, 1]

    # Entropic correction
    F_corr = F + 2.0 * float(temp) * np.log(r)
    F_corr[0] = F[0]

    # Shift tail to zero and scale by kBT
    F_shifted = (
        F_corr - np.mean(F_corr[bin_cut:nb])
    ) / float(temp)

    F_smooth = F_shifted.copy()
    tail_mask = r > r_cut
    F_smooth[tail_mask] = savgol_filter(
        F_shifted[tail_mask], window_length=9, polyorder=2
    )

    return r, F_smooth

Nchains=2
boxsize=240
runs=[1,2,3,4,5,6]
variant='WTARO'
temps = ["35.0","36.0","37.0","38.0","39.0","40.0","42.0","45.0","46.0","48.0","50.0","51.0","52.0","53.0","54.0","55.0","56.0","57.0","58.0","59.0","60.0","61.0","62.0","63.0","64.0","65.0","66.0","67.0","68.0","69.0","70.0"]
equil_dists = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
kspring=5.0
chain_length=137

n_bins=90
bin_cut=40
r_cut = 15.0   # B2 integration cutoff

use_sem = False

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

temps_float = np.array(temps, dtype=float)
norm = colors.Normalize(vmin=temps_float.min(), vmax=temps_float.max())
cmap = cm.viridis

ax.axhline(0, color='k', ls='--',lw=3)

for temp in temps:
    temp_kelvin= np.round(float(temp)*5.6,2)
    pmfs = []
    r_ref = None

    for run in runs:
        fname = f"WHAM_{variant}/N{chain_length}/Free_Energy/free_energy_T_{temp}_run_{run}.txt"
        if not os.path.exists(fname):
            print(f"Missing: T={temp}, run={run}")
            continue

        r, F_smooth = process_pmf(
            fname, temp, n_bins, bin_cut, r_cut
        )

        if r_ref is None:
            r_ref = r
        else:
            # safety check
            if not np.allclose(r, r_ref):
                raise ValueError(f"r-grid mismatch at T={temp}")

        pmfs.append(F_smooth)

    if len(pmfs) < 2:
        print(f"Skipping T={temp}: insufficient runs")
        continue

    pmfs = np.array(pmfs)

    pmf_mean = pmfs.mean(axis=0)
    pmf_err = pmfs.std(axis=0, ddof=1)

    if use_sem:
        pmf_err /= np.sqrt(pmfs.shape[0])

    ax.errorbar(
        r_ref,
        pmf_mean,
        yerr=pmf_err,
        color=cmap(norm(float(temp))),
        lw=1,
        marker='o',
        markersize=3,
        capsize=4,
        alpha=0.9,
        label=rf"$T={temp_kelvin}$"+r'$\;\mathrm{K}$'
    )

ax.axhline(0, color='k', ls='--',lw=3)
ax.set_xlabel(r"$r$")
ax.set_ylabel(r"$\frac{W(r)}{k_BT}$",fontsize=25)
ax.axhline(0, color='k', ls='--',lw=3)
ax.set_xlim([-2, 50])

ax.legend(
    frameon=False,
    loc='lower right',
    ncols=3,
    fontsize=9
)

plt.savefig(
    f"A1LCD_N{chain_length}_PMF_avg_runs_withstd_kelvin.pdf",bbox_inches="tight")


