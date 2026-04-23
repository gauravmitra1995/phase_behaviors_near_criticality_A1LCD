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
import sys
import os
import pandas as pd 
import LASSI_2 as lassi
import BinTrjProc as BR
import TrjProc as TP
import argparse
from mpl_toolkits.mplot3d import Axes3D 
from scipy.spatial import cKDTree
from collections import defaultdict
from collections import deque
import matplotlib.colors as mcolors
import glob
from scipy.optimize import curve_fit

# Set global plotting parameters
plt.rcParams.update({
    'font.family': "sans-serif",
    #'font.family':"DejaVu Sans",
    'font.size': 14,
    'axes.labelsize': 18,
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

def logistic(phi, phi0, k):
    return 1 / (1 + np.exp(-k * (phi - phi0)))

def flatten_2d_list(two_d_list):
    flattened_list = [item for sublist in two_d_list for item in sublist]
    return flattened_list

parser = argparse.ArgumentParser()
parser.add_argument('--Nchains',type=int,default=10000, help="Number of chains.")
args = parser.parse_args()
Nchains = args.Nchains
variant='WTARO'
chain_length=137

T=["59","60","61","62","63","64"]
L=[1000,500,480,450,420,400,360,320,300,280,260,240,220,200]

plt.figure(figsize=(8,4),dpi=300)
cmap = plt.cm.rainbow
norm = mcolors.Normalize(vmin=0, vmax=len(T) - 1) 

percolation_thresholds_fit = {}

for i,temp in enumerate(T):
    densities=[]
    phic=[]
    mean_phic=[]
    std_phic=[]
    for boxsize in L:
        density=np.round((Nchains*chain_length)/(boxsize**3),4)
        pattern = f"./data_phic_Nchains10000_Gaurav6e11/{variant}/clusterresults_T{temp}_L{boxsize}_frame*.npy"
        file_list = glob.glob(pattern)
        all_phic=[]
        densities.append(density)
        for fp in file_list:
            if os.path.exists(fp):
                data = np.load(fp, allow_pickle=True)
                l=np.array(data).tolist()
                dic=dict(l)
                all_phic.append(list(dic['phi_c'])[0])
                all_phic.append(list(dic['phi_c'])[1])
                all_phic.append(list(dic['phi_c'])[2])
            else:
                continue

        mean_phic.append(np.mean(all_phic))
        std_phic.append(np.std(all_phic))

    densities=np.array(densities)
    mean_phic=np.array(mean_phic)

    sorted_indices = np.argsort(densities)
    dens_sorted = densities[sorted_indices]
    phi_c_sorted = mean_phic[sorted_indices]
    
    p0 = [np.median(dens_sorted),1] 
    color = cmap(norm(i))

    popt, pcov = curve_fit(logistic, dens_sorted, phi_c_sorted, p0=p0, bounds=([min(dens_sorted), 0], [max(dens_sorted), np.inf]))
    percolation_thresholds_fit[temp] = np.round(popt[0],5)

    x_fit = np.linspace(min(dens_sorted)-0.1, max(dens_sorted)+0.2, 1000)
    y_fit = logistic(x_fit, *popt)

    plt.errorbar(densities,mean_phic,yerr=std_phic,capsize=8.0,elinewidth=2.0,linestyle='None',marker='o',markersize=6,color=color,ecolor=color,label=r'$T = $'+str(np.round(float(temp)*5.6,2))+r'$\;\mathrm{K}$')
    plt.plot(x_fit, y_fit,color=color,lw=1.0)
    plt.axhline(0.5, color='gray', linestyle='--',lw=2.0)
    plt.axvline(popt[0], color='k', linestyle='--',lw=1.0)

plt.ylabel(r'$f_\mathrm{lc}(\phi)$')
plt.xlabel(r'$\phi$')
plt.ylim(-0.1,1.1)
plt.legend(loc='lower right',ncol=3,fontsize=12)
plt.xlim(0.0,0.2)
plt.text(0.08,0.8,r'$f_\mathrm{lc}(\phi) = \frac{1}{1+\exp[-k(\phi-\phi_p)]}$',fontsize=22)
plt.tight_layout()
plt.savefig('./phic_vs_density_Nchains10000_WTARO.pdf',bbox_inches='tight')

df = pd.DataFrame(list(percolation_thresholds_fit.items()), columns=['Key', 'Value'])
df_indexed = pd.DataFrame.from_dict(percolation_thresholds_fit,orient='index',columns=['Perc. thres'])
data=percolation_thresholds_fit
temps = sorted([float(temp) for temp in data.keys()])
values = []
for t in temps:
    key = str(int(t)) if t.is_integer() else f"{t:.2f}"
    values.append(data[key])
mean=np.mean(values[:]) #after T=58, the perc threshold is nearly temp independent
print("Mean value of the percolation threshold across temperatures: ",mean)

plt.figure(figsize=(8,4),dpi=300)
plt.scatter(np.array(temps)*5.6, values, marker='o',color='green',alpha=0.5)
plt.xlabel(r'$T\;(\mathrm{K})$')
plt.ylabel(r'$\phi_p(T)$')
plt.axhline(mean,ls='--',lw=2.0,color='maroon')
plt.xticks(np.array([59,60,61,62,63,64])*5.6)
plt.ylim(0,0.065)
plt.tight_layout()
plt.yticks([0,0.02,0.04,0.06])
plt.savefig('./phiperc_vs_T_Nchains10000_WTARO.pdf',bbox_inches='tight')
