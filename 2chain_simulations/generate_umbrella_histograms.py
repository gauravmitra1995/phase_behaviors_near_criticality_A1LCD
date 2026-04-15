import os
import sys
import argparse
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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

Nchains=2
boxsize=240
run=1
variant='WTARO'
chain_length=137

temps = ["35.0","36.0", "37.0", "38.0", "39.0", "40.0", "42.0","45.0","46.0","47.0", "48.0", "49.0", "50.0","51.0","52.0","53.0","54.0","55.0","56.0","57.0","58.0","59.0","60.0","61.0","62.0","63.0","64.0","65.0","66.0","67.0","68.0","69.0","70.0"]
equil_dists = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
kspring=5.0

nrows = 11
ncols = 3
fig, axes = plt.subplots(nrows, ncols, figsize=(21,21), dpi=300)
axes = axes.flatten()

for idx, temp in enumerate(temps):

    ax = axes[idx]

    all_vals = []
    for equil_dist in equil_dists:
        
        fname = Path(
            f"{variant}/N{chain_length}/C{Nchains}/L{boxsize}/kspring{kspring}/"
            f"T{temp}/eqdist{equil_dist}/{run}/A1LCD_0_umbrella.dat")

        if not fname.exists():
            print(f"MISSING FILE -> T = {temp}, eq_dist = {equil_dist}")
            continue

        A = np.loadtxt(fname)
        print(temp,equil_dist,A.shape)
        all_vals.append(A[50:, 1])

    all_vals = np.concatenate(all_vals)
    xmin, xmax = np.min(all_vals), np.max(all_vals)
    xgrid = np.linspace(xmin, xmax, 5000)


    cmap = plt.cm.plasma_r
    norm = mcolors.Normalize(vmin=0, vmax=np.array(equil_dists).shape[0] - 1)  # Normalize 
    j=0
    for equil_dist in equil_dists:
        
        c=cmap(norm(j))
        A = np.loadtxt(
            f"{variant}/N{chain_length}/C{Nchains}/L{boxsize}/kspring{kspring}/T{temp}/eqdist{equil_dist}/{run}/A1LCD_0_umbrella.dat"
        )
        vals = A[50:, 1]

        kde = gaussian_kde(vals, bw_method=0.25)
        y = kde(xgrid)
        
        ax.plot(xgrid, y, lw=2.5,color=c)
                
        file_path = f'WHAM_{variant}/N{chain_length}/T_{temp}_r0_{equil_dist}_run_{run}.txt'
        i=1
        with open(file_path, 'w') as file:
            for item in A[50:,1]:
                file.write(f"{i}\t{item}\n")
                i+=1
        j+=1

    ax.set_title(rf"$T = ${np.round(float(temp)*5.6,2)}"+r"$\;\mathrm{K}$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$P(r)$")
    ax.set_ylim(0,0.4)
    ax.set_xlim(0,70)
    ax.set_xticks([0,10,20,30,40,50,60])
    ax.set_yticks([0.0,0.2,0.4])

for i in range(len(temps), len(axes)):
    axes[i].axis("off")

plt.subplots_adjust(wspace=0.5, hspace=2.1)

plt.savefig(f'umbrellahistograms_A1LCD_N{chain_length}_kspring{kspring}_run{run}.pdf')

kspring = 5.0
for temp in temps:
    file_path = f'WHAM_{variant}/N{chain_length}/T_{temp}_run_{run}_metadata.dat'
    with open(file_path, 'w') as file:
        for equil_dist in equil_dists:
            timeseries_file=f'T_{temp}_r0_{equil_dist}_run_{run}.txt'
            file.write(f"{timeseries_file}\t{equil_dist}\t{kspring}\n")
