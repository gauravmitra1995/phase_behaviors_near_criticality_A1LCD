import numpy as np
import matplotlib.pyplot as plt
import glob, os
import sys
import matplotlib.colors as mcolors
from scipy import optimize
import pandas as pd
import seaborn as sns
import pickle
import glob
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("poster")

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

# --- USER INPUTS ---
variant   = "WTARO"             # e.g. 'WT', 'Aro+', 'Aro-'
Nchains   = 10000
chain_length = 137
boxsize   = 240

data_uniform = np.load(f'largestcluster_vs_T_data_{variant}_L{boxsize}.npy',allow_pickle=True)
data_YFR = np.load(f'largestcluster_vs_T_data_{variant}_L{boxsize}_usingYFR.npy',allow_pickle=True)


temperatures = data_uniform[0]
largestclustervalues_uniform = data_uniform[1]
largestclustervalues_YFR = data_YFR[1]

fig,ax=plt.subplots(figsize=(5.5,3.5),dpi=300)
To=53.58
Tp=58.50
Tc=59.36

ax.plot(temperatures,np.array(largestclustervalues_uniform)/Nchains,marker='o',ms=8,ls='None',lw=2.0,color='maroon',label='Uniform (all residues)')
ax.plot(temperatures,np.array(largestclustervalues_YFR)/Nchains,marker='o',ms=8,ls='None',lw=2.0,color='tomato',label='Sticker-mediated')
ax.set_ylabel(r'$f_\mathrm{lc}$')
ax.set_xlabel(r'$T\;(\mathrm{K})$')
ax.set_xticks(np.array([50,52,54,56,58,60])*5.6)
ax.set_ylim(bottom=0.8)
ax.set_ylim(top=1.01)
ax.legend(loc='best',ncol=1,fontsize=10)
plt.tight_layout()
plt.savefig(f'largestclustersize_vs_T_plot.pdf',bbox_inches='tight')

