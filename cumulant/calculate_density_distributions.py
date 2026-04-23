import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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
import matplotlib.colors as mcolors
import scipy.stats as stats
import glob

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

parser = argparse.ArgumentParser()
parser.add_argument('--temp',type=str, required=True, help="Temp value.")
parser.add_argument('--boxsize',type=int,default=240, help="Box size.")

args = parser.parse_args()
temp = args.temp
boxsize = args.boxsize

Nchains=10000
L=[size for size in range(60, 191, 10)]
Lbox=boxsize
variant='WTARO'
Nsubboxes=10000
cmap = plt.cm.rainbow_r

norm = mcolors.Normalize(vmin=0, vmax=len(L) - 1)

global_max = 0  
histograms = {}

for Lsub in L:
    pattern = f'Density_subbox_Nchains{Nchains}_6e11_last100snaps/{variant}/Lbox{Lbox}/Density_T{temp}_L{Lsub}_frame*_Nsubboxes{Nsubboxes}.npy'
    file_list = glob.glob(pattern)
    densities_frames = []
    for fp in file_list:
        densities = np.load(fp, allow_pickle=True)
        densities_frames.append(densities)
    if not densities_frames:
        continue
    densities_frames = np.array(densities_frames).flatten()
    counts, bins = np.histogram(densities_frames, bins=200, density=True)
    histograms[Lsub] = (counts, bins)
    global_max = max(global_max, np.max(counts))

print("Global max is:", global_max)

fig, ax = plt.subplots(figsize=(7.5,4.5), dpi=300)

for idx, Lsub in enumerate(L):
    if Lsub not in histograms:
        continue
    counts,bins = histograms[Lsub]
    counts = counts / global_max
    color = cmap(norm(idx))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    print(Lsub,np.min(bin_centers),np.max(bin_centers))
    ax.plot(bin_centers, counts, linestyle='-', linewidth=1.5, marker='o',markersize=3.0,color=color, label=r'$L = $'+f'{Lsub}')

ax.legend(loc='upper right',ncol=2,fontsize=10)
ax.set_title(r'$T = $' + str(np.round(float(temp)*5.6,2))+r'$\;\mathrm{K}$')
ax.set_ylabel(r"$p(\phi)$")
ax.set_xlabel(r"$\phi$")
ax.set_ylim(0, 1)
plt.savefig(f'densitydistribution_T{temp}_subboxsize{Lbox}.pdf', bbox_inches='tight')



