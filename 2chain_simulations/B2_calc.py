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
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

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

To=53.58
Tc=59.36

Nchains=2
boxsize=240
chain_length=137
seeds=[1,2,3,4,5,6]
variant='WTARO'
temps = ["35.0","36.0","37.0","38.0","39.0","40.0","42.0","45.0","46.0","47.0","48.0","49.0","50.0","51.0","52.0","53.0","54.0","55.0","56.0","57.0","58.0","59.0","60.0","61.0","62.0","63.0","64.0","65.0","66.0","67.0","68.0","69.0","70.0"]
equil_dists = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
kspring=5.0

n_bins = 60
bin_cut = 40
r_cut = 15.0

n_sub = 12     # PMF points per integration
n_boot = 10000   # subsampling realizations
rng_seed = 42
rng = np.random.default_rng(rng_seed)


PMFs = {}       
r_vals = {}        
tail_means = []

for temp in temps:

    PMFs[temp] = {}
    r_vals[temp] = {}

    for seed in seeds:

        Free_energy = np.loadtxt(
            f"WHAM_WTARO/N{chain_length}/Free_Energy/free_energy_T_{temp}_run_{seed}.txt",
            skiprows=1
        )

        nb = min(n_bins, Free_energy.shape[0])
        r = Free_energy[:nb, 0]
        F = Free_energy[:nb, 1]

        # --- Jacobian correction ---
        Corrected_free_energy = F + 2.0 * float(temp) * np.log(r)
        Corrected_free_energy[0] = F[0]

        # Reduce PMF
        F_shifted = (
            Corrected_free_energy
            - np.mean(Corrected_free_energy[bin_cut:nb])
        ) / float(temp)

        F_smooth = F_shifted.copy()
        tail_mask = r > r_cut
        F_smooth[tail_mask] = savgol_filter(
            F_shifted[tail_mask],
            window_length=9,
            polyorder=2
        )

        PMFs[temp][seed] = F_smooth
        r_vals[temp][seed] = r

        # collect tail mean for global reference
        tail_means.append(np.mean(F_smooth[tail_mask]))


global_tail_mean = np.mean(tail_means)

print("Global PMF tail reference:", global_tail_mean)

B2_by_T = {}      # B2_by_T[T] = list of B2 values (one per seed)

for temp in temps:

    B2_by_T[temp] = []

    for seed in seeds:

        Free_energy = np.loadtxt(
            f"WHAM_WTARO/N{chain_length}/Free_Energy/free_energy_T_{temp}_run_{seed}.txt",
            skiprows=1
        )

        nb = min(n_bins, Free_energy.shape[0])
        r = Free_energy[:nb, 0]
        F = Free_energy[:nb, 1]

        Corrected_free_energy = F + 2.0 * float(temp) * np.log(r)
        Corrected_free_energy[0] = F[0]

        F_shifted = (
            Corrected_free_energy
            - np.mean(Corrected_free_energy[bin_cut:nb])
        ) / float(temp)

        F_smooth = F_shifted.copy()
        tail_mask = r > r_cut
        F_smooth[tail_mask] = savgol_filter(
            F_shifted[tail_mask],
            window_length=9,
            polyorder=2
        )

        F_aligned = F_smooth - (np.mean(F_smooth[tail_mask]) - global_tail_mean)
 
        phys_mask = (r <= r_cut)

        r_phys = r[phys_mask]
        F_phys = F_aligned[phys_mask]

        n_pts = len(r_phys)
        if n_sub >= n_pts:
            raise ValueError("n_sub must be smaller than number of points within r_cut")

        B2_boot = np.zeros(n_boot)

        for i in range(n_boot):

            # always keep endpoints
            inner_idx = np.arange(1, n_pts - 1)

            chosen = rng.choice(
                inner_idx,
                size=n_sub - 2,
                replace=False
            )

            sel_idx = np.sort(
                np.concatenate(([0], chosen, [n_pts - 1]))
            )

            r_sel = r_phys[sel_idx]
            F_sel = F_phys[sel_idx]

            integrand = (np.exp(-F_sel) - 1.0) * r_sel**2

            B2_boot[i] = -2.0 * np.pi * np.trapezoid(
                integrand,
                r_sel
            )

        B2_seed_mean = np.mean(B2_boot)
        
        B2_by_T[temp].append(B2_seed_mean)
    
    #print(temp,B2_by_T[temp])
temps_arr = np.array(temps, dtype=float)

B2_mean_arr = np.array([
    np.mean(B2_by_T[T]) for T in temps
])

B2_std_arr = np.array([
    np.std(B2_by_T[T], ddof=1) for T in temps
])


#Spline Interpolation 
start = 7
T_sub  = temps_arr[start:]
B2_sub = B2_mean_arr[start:]
err_sub = B2_std_arr[start:]

weights = 1.0 / err_sub**2

spline = UnivariateSpline(
    T_sub,
    B2_sub,
    w=weights,
    s=len(T_sub),k=5
)
T_fine = np.linspace(T_sub.min(), T_sub.max(), 10000)
B2_fine = spline(T_fine)

sign_changes = np.where(np.diff(np.sign(B2_fine)))[0]

if len(sign_changes) == 0:
    raise RuntimeError("No zero crossing found in spline")

i0 = sign_changes[0]

T_boyle_spline = np.interp(
    0.0,
    [B2_fine[i0], B2_fine[i0+1]],
    [T_fine[i0], T_fine[i0+1]]
)

print(f"Boyle temperature (spline interpolation): {T_boyle_spline:.2f}")


#Linear Interpolation
sign_changes = np.where(np.diff(np.sign(B2_mean_arr)))[0]

if len(sign_changes) == 0:
    raise RuntimeError("No B2 sign change found")

i = sign_changes[0]

T1, T2 = temps_arr[i], temps_arr[i+1]
B1, B2 = B2_mean_arr[i], B2_mean_arr[i+1]

T_boyle = T1 + (0.0 - B1) * (T2 - T1) / (B2 - B1)
print(f"Boyle temperature (linear interpolation): {T_boyle:.2f}")

B2_max=B2_sub[-1]
B2_mean_norm= B2_sub/B2_max
B2_std_norm  = err_sub/B2_max

print(B2_max)
print(B2_mean_norm)

plt.figure(figsize=(8,6), dpi=300)
plt.errorbar(
    T_sub*5.6,
    B2_mean_norm,
    yerr=B2_std_norm,
    fmt='o',
    ms=6,
    color='k',
    ecolor='k',
    elinewidth=1,
    capsize=7)

plt.axvline(To*5.6,color='maroon',ls=':',lw=2,label=r'$T^\ast\;=\;$'+str(np.round(To*5.6,2))+r'$\;\mathrm{K}$')
plt.axvline(Tc*5.6,color='darkorange',ls=':',lw=2,label=r'$T_c\;=\;$'+str(np.round(Tc*5.6,2))+r'$\;\mathrm{K}$')
plt.axvline(T_boyle*5.6, color='purple', ls=':', lw=2,label=r'$T_\theta\;=\;$'+str(np.round(T_boyle*5.6,2))+r'$\;\mathrm{K}$')
plt.axhline(0, color='crimson', ls='--',lw=2,dashes=(2,2),label=r'$B_{2}^{\prime}(T)\;=\;0$')
plt.scatter(T_boyle*5.6,0,color='purple',s=100,marker='x',ls='None')
plt.xlabel(r"$T\;(\mathrm{K})$")
plt.ylabel(r"$B_{2}^{\prime}(T)\;=\;\frac{B_2(T)}{B_2^\mathrm{max}}$",fontsize=25)
plt.yticks([-16,-14,-12,-10,-8,-6,-4,-2,0,2])
plt.xticks(np.array([45,50,55,60,65,70,75])*5.6)
plt.legend(frameon=False,loc='lower right',ncol=1,fontsize=12.5)
plt.savefig("B2_vs_T_plot_A1LCD.pdf", bbox_inches="tight")
