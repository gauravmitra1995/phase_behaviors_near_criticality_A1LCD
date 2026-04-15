import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
#from skimage import measure
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("poster")
import pickle
import sys
import os
import LASSI_2 as lassi
import BinTrjProc as BR
import TrjProc as TP
import argparse
import glob
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from collections import defaultdict
from collections import deque
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import BoundaryNorm
from scipy.stats import gaussian_kde

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

def unwrap_chain(chain, boxsize):
    chain = np.asarray(chain, dtype=float)
    unwrapped = np.zeros_like(chain)
    unwrapped[0] = chain[0]
    for i in range(1, len(chain)):
        dr = chain[i] - chain[i-1]
        dr -= boxsize * np.round(dr / boxsize)
        unwrapped[i] = unwrapped[i-1] + dr
    return unwrapped

def compute_Rij_map(chains, boxsize):
    chains = [np.asarray(c, dtype=float) for c in chains]
    nres = chains[0].shape[0]
    Rij2_accum = np.zeros((nres, nres), dtype=float)
    n_chains = len(chains)

    Rg_values = []
    Ree_values = []

    for chain in chains:
        chain_unwrapped = unwrap_chain(chain, boxsize)

        # --- Rij^2 calculation (CRITICAL FIX) ---
        diff = chain_unwrapped[:, None, :] - chain_unwrapped[None, :, :]
        Rij2 = np.sum(diff**2, axis=-1)   # <-- r^2, NOT |r|
        Rij2_accum += Rij2

        # --- Rg ---
        com = np.mean(chain_unwrapped, axis=0)
        dr = chain_unwrapped - com
        Rg = np.sqrt(np.mean(np.sum(dr**2, axis=1)))
        Rg_values.append(Rg)

        # --- Ree ---
        Ree = np.sqrt(np.sum((chain_unwrapped[-1] - chain_unwrapped[0])**2))
        Ree_values.append(Ree)

    Rij2_mean = Rij2_accum / n_chains

    return Rij2_mean, np.array(Rg_values), np.array(Ree_values)

def extract_Rs_from_Rij2(Rij2_mean):
    nres = Rij2_mean.shape[0]
    s_vals = np.arange(1, nres)
    R_s = np.zeros_like(s_vals, dtype=float)

    for idx, s in enumerate(s_vals):
        vals = [Rij2_mean[i, i+s] for i in range(nres - s)]
        R_s[idx] = np.sqrt(np.mean(vals)) 

    return s_vals, R_s


def compute_Rs_vs_T(temps1, temps2, variant, boxsize, Nchains, seeds, n_equil):

    Rs_vs_T = {}
    Rg_vs_T = {}
    Rgerror_vs_T={}
    Ree_vs_T = {}
    Reeerror_vs_T={}

    for temp in temps1:

        Rij2_accum_T = None
        Rg_all = []
        Ree_all = []
        n_samples = 0

        for seed in seeds:

            Extractor_obj = BR.TrjExtractor(
                f'/project/fava/work/minafarag/A1LCD/Spacer_Production_NQT_2_Single/'
                f'No_NCPR_Term/E0/{variant}/R1/T{temp}/B{boxsize}/P{boxsize}/'
                f'C{Nchains}/{seed}/A1LCD_trj.lassi'
            )

            BC = Extractor_obj.extract_coords()
            nframes = BC.shape[0]

            for frame in range(nframes - n_equil, nframes):

                Rij2, Rg_vals, Ree_vals = compute_Rij_map([BC[frame]], boxsize)

                if Rij2_accum_T is None:
                    Rij2_accum_T = np.zeros_like(Rij2)

                Rij2_accum_T += Rij2
                Rg_all.extend(Rg_vals)
                Ree_all.extend(Ree_vals)
                n_samples += 1

        Rij2_mean_T = Rij2_accum_T / n_samples
        s_vals, R_s = extract_Rs_from_Rij2(Rij2_mean_T)

        Rs_vs_T[temp] = (s_vals, R_s)
        Rg_vs_T[temp] = np.mean(Rg_all)
        Rgerror_vs_T[temp]=np.std(Rg_all)
        Ree_vs_T[temp] = np.mean(Ree_all)
        Reeerror_vs_T[temp]=np.std(Ree_all)

    for temp in temps2:

        Rij2_accum_T = None
        Rg_all = []
        Ree_all = []
        n_samples = 0

        for seed in seeds:

            Extractor_obj = BR.TrjExtractor(
                f'/project/fava/work/gmitra/A1LCD/Spacer_Production_NQT_2_Single/'
                f'No_NCPR_Term/{variant}/C{Nchains}/L{boxsize}/T{temp}/{seed}/A1LCD_trj.lassi'
            )

            BC = Extractor_obj.extract_coords()
            nframes = BC.shape[0]

            for frame in range(nframes - n_equil, nframes):

                Rij2, Rg_vals, Ree_vals = compute_Rij_map([BC[frame]], boxsize)

                if Rij2_accum_T is None:
                    Rij2_accum_T = np.zeros_like(Rij2)

                Rij2_accum_T += Rij2
                Rg_all.extend(Rg_vals)
                Ree_all.extend(Ree_vals)
                n_samples += 1

        Rij2_mean_T = Rij2_accum_T / n_samples
        s_vals, R_s = extract_Rs_from_Rij2(Rij2_mean_T)

        Rs_vs_T[temp] = (s_vals, R_s)
        Rg_vs_T[temp] = np.mean(Rg_all)
        Rgerror_vs_T[temp]=np.std(Rg_all)
        Ree_vs_T[temp] = np.mean(Ree_all)
        Reeerror_vs_T[temp]=np.std(Ree_all)

    return Rs_vs_T, Rg_vs_T, Rgerror_vs_T, Ree_vs_T, Reeerror_vs_T

def extract_nu_from_Rs(s, R_s, smin, smax):
    mask = (s >= smin) & (s <= smax)

    log_s = np.log(s[mask])
    log_R = np.log(R_s[mask])

    nu, intercept = np.polyfit(log_s, log_R, 1)
    return nu

def compute_nu(s_values, R_values, s_l, s_u):
    mask = (s_values >= s_l) & (s_values <= s_u)
    s_sel, R_sel = s_values[mask], R_values[mask]
    nu_list = []
    for i in range(len(s_sel)):
        for j in range(i+1, len(s_sel)):
            sx, sy = s_sel[i], s_sel[j]
            Rx, Ry = R_sel[i], R_sel[j]
            if Ry > Rx:
                num = np.log(Ry) - np.log(Rx)
                den = np.log(sy) - np.log(sx)
                if np.isfinite(num) and den != 0:
                    nu_list.append(num / den)

    nu_all = np.array(nu_list)
    nu_mean = np.mean(nu_all)
    nu_std = np.std(nu_all)
    return nu_all, nu_mean, nu_std

boxsize=120
Nchains=1
chain_length=137
variant='WTARO'
seeds=[1,2,3]
temps1=[5.0,10.0,15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0]
temps2=[7.0,8.0,12.0,14.0,17.0,18.0,22.0,24.0,27.0,28.0,32.0,34.0,37.0,38.0,42.0,44.0,47.0,48.0]

Rs_vs_T, Rg_vs_T, Rgerror_vs_T, Ree_vs_T, Reeerror_vs_T = compute_Rs_vs_T(temps1, temps2, variant, boxsize, Nchains, seeds, 20000)

#R(s) vs s plot 

plt.figure(figsize=(8,6), dpi=300)

Ts = np.array(sorted(Rs_vs_T.keys()))
colors = cm.viridis(np.linspace(0, 1, len(Ts)))

for color, T in zip(colors, Ts):
    temp_K= np.round(float(T)*5.6,2)
    s, R_s = Rs_vs_T[T]
    plt.plot(
        s, R_s,
        'o-',
        lw=0.75,
        markersize=2.3,
        color=color,
        #label=rf"$T = {T}$"
        label=rf"$T={temp_K}$ "+r'$\mathrm{K}$'
    )

plt.xlabel(r"$s = \mid j - i \mid$")
plt.ylabel(r"$\langle R(s) \rangle$")
plt.legend(loc='upper left', ncol=3, fontsize=7.5)
plt.tight_layout()
plt.savefig("Rij_vs_s_plot_variousT_A1LCD.pdf", bbox_inches="tight")

nu_vs_T_mean = {}
nu_vs_T_error = {}
nu_vs_T_all={}
s_l=40
s_u=120

for T, (s, R_s) in sorted(Rs_vs_T.items()):
    #nu_vs_T[T] = extract_nu_from_Rs(s,R_s,40,120)
    nu_all, nu_mean, nu_std = compute_nu(s, R_s, s_l, s_u)
    nu_vs_T_mean[T]= nu_mean
    nu_vs_T_error[T]= nu_std
    nu_vs_T_all[T]= nu_all
    
Ts = np.array(sorted(nu_vs_T_mean.keys()))
nus = np.array([nu_vs_T_mean[T] for T in Ts])
error_nus = np.array([nu_vs_T_error[T] for T in Ts])

Ts = Ts[Ts>25.0]
nus = np.array([nu_vs_T_mean[T] for T in Ts])
error_nus = np.array([nu_vs_T_error[T] for T in Ts])

Tc=59.36*5.6 #in Kelvin

from scipy.interpolate import interp1d

plt.figure(figsize=(8,6), dpi=300)

# cubic interpolation (smooth but stable)
nu_interp = interp1d(Ts, nus, kind='cubic', fill_value='extrapolate')

# fine temperature grid
T_fine = np.linspace(Ts.min(), Ts.max(), 5000)
nu_fine = nu_interp(T_fine)

T_theta = T_fine[np.argmin(np.abs(nu_fine - 0.5))]
print("Interpolated apparent T_theta =", T_theta)

To=53.58

plt.errorbar(Ts*5.6, nus, yerr=error_nus, marker='o',ms=8, ls='None', capsize=7, elinewidth=1.5, ecolor='navy',color='navy')
#plt.plot(T_fine, nu_fine, marker='None',ls='-', lw=1.0, color='navy')
plt.axvline(To*5.6,color='maroon',ls='--',lw=2,label=r'$T^\ast = $'+str(np.round(To*5.6,2))+r'$\;\mathrm{K}$')
plt.axvline(Tc,ls='--',c='darkorange',lw=2,label=r'$T_c = $'+str(np.round(Tc,2))+r'$\;\mathrm{K}$')
plt.axvline(T_theta*5.6,ls='--',c='crimson',lw=2,label=r'$T_{\theta,\mathrm{app}} = $'+str(np.round(T_theta*5.6,2))+r'$\;\mathrm{K}$')
plt.axhline(0.5, ls=':', c='crimson', lw=4, dashes=(1,1), label=r'$\nu\;(T) = 0.5$')

plt.yticks([0.30,0.35,0.4,0.45,0.5,0.55])
plt.xticks(np.array([25.0,30.0,35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0])*5.6)
plt.xlabel(r"$T\;(\mathrm{K})$")
plt.ylabel(r"$\nu\;(T)$")
plt.legend(loc='lower right',fontsize=10)
plt.tight_layout()
plt.savefig('nu_vs_T_A1LCD.pdf',bbox_inches='tight')


## Rg and Ree vs T

Ts = np.array(sorted(Rg_vs_T.keys()))
Rgs = np.array([Rg_vs_T[T] for T in Ts])
Rg_errors = np.array([Rgerror_vs_T[T] for T in Ts])

plt.figure(figsize=(8,6), dpi=300)
plt.errorbar(Ts*5.6, Rgs, yerr=Rg_errors,marker='o',ls='-',lw=1.3, markersize=6,color='purple',elinewidth=0.8,capsize=6.5,ecolor='purple')

plt.xlabel(r"$T\;(\mathrm{K})$")
plt.ylabel(r"$R_g$ (l.u.)")
plt.ylim(2,10)
plt.xticks(np.array([0,10,20,30,40,50,60,70])*5.6)
plt.tight_layout()
plt.savefig("Rg_vs_T_A1LCD.pdf", bbox_inches="tight")



Ts = np.array(sorted(Ree_vs_T.keys()))
Rees = np.array([Ree_vs_T[T] for T in Ts])
Ree_errors = np.array([Reeerror_vs_T[T] for T in Ts])

plt.figure(figsize=(8,6), dpi=300)
plt.errorbar(Ts*5.6, Rees, yerr=Ree_errors,marker='o',ls='-',lw=1.3, markersize=6,color='darkgreen',elinewidth=0.8,capsize=6.5,ecolor='darkgreen')

plt.xlabel(r"$T\;(\mathrm{K})$")
plt.ylabel(r"$R_{ee}$ (l.u.)")
plt.ylim(0,30)
plt.xticks(np.array([0,10,20,30,40,50,60,70])*5.6)
plt.tight_layout()
plt.savefig("Ree_vs_T_A1LCD.pdf", bbox_inches="tight")
