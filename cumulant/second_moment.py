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
import scipy as sp
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

Nchains=10000
variant='WTARO'
cmap = plt.cm.plasma_r
Nsubboxes=10000

def second_moment(rho_b,mean_rho):
    first_moment=[(ele-mean_rho)**1 for ele in rho_b]
    second_moment=[(ele-mean_rho)**2 for ele in rho_b]
    fourth_moment=[(ele-mean_rho)**4 for ele in rho_b]
    mean_first_moment=np.mean(first_moment)
    mean_second_moment=np.mean(second_moment)
    mean_fourth_moment=np.mean(fourth_moment)
    deno=mean_second_moment*mean_second_moment
    binderratio=(mean_fourth_moment/deno)
    bc=1-(binderratio/3)
    return(second_moment)

def bootstrap_binder(data,n_bootstrap,mean_rho):
    bootstrapped_values = []
    N = len(data)
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=N, replace=True)  
        a = second_moment(resample,mean_rho)
        bootstrapped_values.append(a)

    mean_binder = np.mean(bootstrapped_values)  
    std_binder = np.std(bootstrapped_values)  
    return mean_binder, std_binder

L=[size for size in range(160, 191, 10)]
norm = mcolors.Normalize(vmin=0, vmax=len(L) - 1)
T=["50","52","54","55","56","57","58","58.50","58.75","59","59.25","59.50","59.75","60","61","62","64"]
Nchains=10000
Lsim=240
Nsubboxes=10000
mean_rho=float((Nchains*137)/(Lsim**3))
fig=plt.figure(figsize=(7.5,4.5),dpi=300)

from time import perf_counter
t0 = perf_counter()

for idx,L_box in enumerate(L):
    Binder_para,Temp_binder,Binder_err,fourth_T,mean_T,second_T=[],[],[],[],[],[]
    for i,temp in enumerate(T):
        pattern= f'Density_subbox_Nchains{Nchains}_6e11_last100snaps/{variant}/Lbox{Lsim}/Density_T{temp}_L{L_box}_frame*_Nsubboxes{Nsubboxes}.npy'
        file_list = glob.glob(pattern)
        densities_frames=[]
        for fp in file_list:
            densities = np.load(fp,allow_pickle=True)
            densities_frames.append(densities)
        densities_frames=np.array(densities_frames)
        densities_frames=densities_frames.flatten()

        #Apply Bootstrapping
        binder_mean, binder_std = bootstrap_binder(densities_frames,1,mean_rho)
        Binder_para.append(binder_mean)
        Binder_err.append(binder_std)
        Temp_binder.append(float(temp))
    print("Box size "+str(L_box)+" processed.....")
    color = cmap(norm(idx))
    plt.errorbar(np.array(Temp_binder)*5.6,Binder_para,yerr=Binder_err,marker='o',markersize=8,capsize=10,color=color,lw=1.8,elinewidth=1.3,label=r'$L = $'+str(L_box))

plt.ylabel(r'$\Delta\phi_{L}^2 (T)$')
plt.xlabel(r'$T\;(\mathrm{K})$')
plt.legend(loc='upper left',ncol=2)
plt.xticks(np.array([50.0,52.0,54.0,56.0,58.0,60.0,62.0,64.0])*5.6)
plt.yticks([0.0,0.01,0.02,0.03,0.04,0.05])
plt.tight_layout()
plt.savefig('second_moment_plot_boxsize'+str(Lsim)+'.pdf',bbox_inches='tight')

elapsed = perf_counter() - t0
print(f"Elapsed: {elapsed:.3f} s")
