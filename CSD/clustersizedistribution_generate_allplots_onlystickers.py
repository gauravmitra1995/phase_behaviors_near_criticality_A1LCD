import numpy as np
from skimage import measure
import glob
import matplotlib.pyplot as plt
import pickle
import sys
import os
import argparse
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from collections import defaultdict
from collections import deque
from collections import Counter
from scipy.optimize import curve_fit
import glob
import sklearn
from sklearn.utils import resample

import LASSI_2 as lassi
import BinTrjProc as BR
import TrjProc as TP

import seaborn as sns

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

# Function to average distributions across seeds

def average_distribution(list_of_dicts):
    all_sizes = set()
    for dist in list_of_dicts:
        all_sizes.update(dist.keys())
    avg = {}
    for size in all_sizes:
        vals = [dist.get(size, 0.0) for dist in list_of_dicts]
        avg[size] = sum(vals) / len(list_of_dicts)
    return dict(sorted(avg.items()))

def compute_derivative(sizes, probs):
    n = np.array(sizes, dtype=float)
    P = np.array(probs, dtype=float)
    # Use numpy.gradient for central differences
    deriv = np.gradient(P, n)
    return deriv

def compute_loglog_derivative(sizes, probs):
    log_n = np.log10(sizes.astype(float))
    log_P = np.log10(probs.astype(float))
    deriv = np.gradient(log_P, log_n)
    return deriv

def flatten_2d_list(two_d_list):
    flattened_list = [item for sublist in two_d_list for item in sublist]
    return flattened_list

variant='WTARO'
Nchains=10000
chain_length=137
boxsize=240
tau_values=[]
tau_error_values=[]
temp_values=[]
Tc=59.36

import matplotlib.cm as cm

# --- define all temperatures once ---
all_T = ["50","52","54","55","56","57","58","58.50","58.75","59","59.25"]

# --- assign unique colors ---
color_list = cm.tab20(np.linspace(0, 1, len(all_T)))
color_dict = dict(zip(all_T, color_list))

# --- markers (reused safely) ---
markers = ['o', '^', 's', 'p', '*', 'X', 'D']


largestcluster_values=[]


T=["50","52"]

# --- store results ---
n_tol_dict = {}
slope_tol_dict = {}

for i,temp in enumerate(T):
    density=np.round((Nchains*chain_length)/(boxsize**3),4)
    pattern=f"./data_CSD_Nchains{Nchains}_Gaurav6e11_usingYFR/{variant}/CSDresults_T{temp}_L{boxsize}_frame*.npy"
    file_list=glob.glob(pattern)
    dense_phase_CSD=[]
    dilute_phase_CSD=[]
    for fp in file_list:
        if os.path.exists(fp):
            data = np.load(fp, allow_pickle=True)
            l=data.tolist()
            dense_phase_CSD.append(l['Dense_phase_CSD'])
            dilute_phase_CSD.append(l['Dilute_phase_CSD'])


    dense_phase_CSD=flatten_2d_list(dense_phase_CSD)
    dilute_phase_CSD=flatten_2d_list(dilute_phase_CSD)
    P_dense_seeds= dense_phase_CSD
    P_dilute_seeds= dilute_phase_CSD

    # Compute averaged distributions
    P_avg_dilute = average_distribution(P_dilute_seeds)

    # Prepare data for plotting
    sizes_d = np.array(list(P_avg_dilute.keys()))
    probs_d = np.array(list(P_avg_dilute.values()))

    dP_dn = compute_derivative(sizes_d, probs_d)

    dlogP_dlogn = compute_loglog_derivative(sizes_d, probs_d)
    curvature = np.gradient(dlogP_dlogn,np.log10(sizes_d))
    slope=dlogP_dlogn


# --- distinct markers for each T ---
markers = ['o', '^','s','p','*','X','D']  # reuse if more T

fig,ax1 = plt.subplots(figsize=(5.5, 3.5), dpi=300)

for i,temp in enumerate(T):

    marker = markers[i]
    color = color_dict[temp]
    
    density=np.round((Nchains*chain_length)/(boxsize**3),4)
    pattern=f"./data_CSD_Nchains{Nchains}_Gaurav6e11_usingYFR/{variant}/CSDresults_T{temp}_L{boxsize}_frame*.npy"
    file_list=glob.glob(pattern)
    dense_phase_CSD=[]
    dilute_phase_CSD=[]
    for fp in file_list:
        if os.path.exists(fp):
            data = np.load(fp, allow_pickle=True)
            l=data.tolist()
            dense_phase_CSD.append(l['Dense_phase_CSD'])
            dilute_phase_CSD.append(l['Dilute_phase_CSD'])
            
    dense_phase_CSD=flatten_2d_list(dense_phase_CSD)
    dilute_phase_CSD=flatten_2d_list(dilute_phase_CSD)
    P_dense_seeds= dense_phase_CSD
    P_dilute_seeds= dilute_phase_CSD

    # Compute averaged distributions
    P_avg_dilute = average_distribution(P_dilute_seeds)

    total = sum(P_avg_dilute.values())
    print(f"T = {temp}: sum of averaged P(n) = {total:.6f}")

    # Prepare data for plotting
    sizes_d = np.array(list(P_avg_dilute.keys()))
    probs_d = np.array(list(P_avg_dilute.values()))

    # Dense phase: report cluster sizes, not a distribution
    dense_sizes = [list(d.keys())[0] for d in dense_phase_CSD]
    dense_mean = np.mean(dense_sizes)
    dense_std = np.std(dense_sizes) 

    # Plot on first axis (1 ≤ n ≤ 100)
    temperature=float(temp)*5.6
    
    ax1.loglog(sizes_d, probs_d, marker=marker, linestyle='None', lw=1.0,markersize=5,color=color,alpha=0.7,label=r'$T = $'+f"{temperature:.2f}"+r'$\;\mathrm{K}$')
    #ax1.loglog(sizes_d, probs_d, marker=marker, linestyle='None', lw=1.0,markersize=5,color=color,alpha=0.7,label=r'$T = $'+f"{temperature:.2f}"+r'$\;\mathrm{K}$'+' , '+r'$\tau = $'+f"{tau_rounded:.2f}")
    #ax1.loglog(sizes_d, P_fit_d,linestyle='--',color='k',lw=0.7)
    
    temp_values.append(temp)
    #secondlargestcluster_values.append(sizes_d[-1])
    largestcluster_values.append(np.median(sorted(dense_sizes)))
    
    #ax1.set_xlim(0, 1e5)
    #ax1.set_xticks([1, 10, 100, 1000, 10000, 100000])
    ax1.set_xlim(0, 1e2)
    ax1.set_xticks([1, 10, 100])
    ax1.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])
    
    #ax1.set_ylim(1e-5,10)
    ax1.legend(loc='upper right',fontsize=9)
    ax1.yaxis.tick_left()
    ax1.set_ylabel(r'$p(n)$')
    ax1.set_xlabel(r'$n$')


    #ax1.loglog(dense_sizes, [1.0]*len(dense_sizes), marker=marker, linestyle='None',lw=1.0, markersize=3, color='red', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'CSDplot_{variant}_L{boxsize}_Regime1_usingYFR.pdf',bbox_inches='tight')


T=["54","55","56","57","58"]

# --- store results ---
n_tol_dict = {}
slope_tol_dict = {}

for i,temp in enumerate(T):
    density=np.round((Nchains*chain_length)/(boxsize**3),4)
    pattern=f"./data_CSD_Nchains{Nchains}_Gaurav6e11_usingYFR/{variant}/CSDresults_T{temp}_L{boxsize}_frame*.npy"
    file_list=glob.glob(pattern)
    dense_phase_CSD=[]
    dilute_phase_CSD=[]
    for fp in file_list:
        if os.path.exists(fp):
            data = np.load(fp, allow_pickle=True)
            l=data.tolist()
            dense_phase_CSD.append(l['Dense_phase_CSD'])
            dilute_phase_CSD.append(l['Dilute_phase_CSD'])

    dense_phase_CSD=flatten_2d_list(dense_phase_CSD)
    dilute_phase_CSD=flatten_2d_list(dilute_phase_CSD)
    P_dense_seeds= dense_phase_CSD
    P_dilute_seeds= dilute_phase_CSD

    # Compute averaged distributions
    P_avg_dilute = average_distribution(P_dilute_seeds)

    # Prepare data for plotting
    sizes_d = np.array(list(P_avg_dilute.keys()))
    probs_d = np.array(list(P_avg_dilute.values()))

    dP_dn = compute_derivative(sizes_d, probs_d)

    dlogP_dlogn = compute_loglog_derivative(sizes_d, probs_d)
    curvature = np.gradient(dlogP_dlogn,np.log10(sizes_d))
    slope=dlogP_dlogn

    window = 5   # try 3–7 depending on smoothness
    kernel = np.ones(window) / window
    smooth_slope = np.convolve(slope, kernel, mode='valid')
    sizes_smooth = sizes_d[window-1:]

    ref = np.median(slope[:5])

    threshold_frac = 0.5
    n_tol = sizes_d[-1]

    for i in range(len(smooth_slope)):
        if abs(smooth_slope[i] - ref) > threshold_frac * abs(ref):
            n_tol = sizes_smooth[i]
            break

    print(f"Reference slope = {ref}, T={temp}, n_tol={n_tol}")
    n_tol_dict[temp] = n_tol

# --- distinct markers for each T ---
markers = ['o', '^','s','p','*','X','D']  # reuse if more T

fig,ax1 = plt.subplots(figsize=(5.5, 3.5), dpi=300)

for i,temp in enumerate(T):

    marker = markers[i]
    color = color_dict[temp]
    
    density=np.round((Nchains*chain_length)/(boxsize**3),4)
    pattern=f"./data_CSD_Nchains{Nchains}_Gaurav6e11_usingYFR/{variant}/CSDresults_T{temp}_L{boxsize}_frame*.npy"
    file_list=glob.glob(pattern)
    dense_phase_CSD=[]
    dilute_phase_CSD=[]
    for fp in file_list:
        if os.path.exists(fp):
            data = np.load(fp, allow_pickle=True)
            l=data.tolist()
            dense_phase_CSD.append(l['Dense_phase_CSD'])
            dilute_phase_CSD.append(l['Dilute_phase_CSD'])
            
    dense_phase_CSD=flatten_2d_list(dense_phase_CSD)
    dilute_phase_CSD=flatten_2d_list(dilute_phase_CSD)
    P_dense_seeds= dense_phase_CSD
    P_dilute_seeds= dilute_phase_CSD

    # Compute averaged distributions
    P_avg_dilute = average_distribution(P_dilute_seeds)

    total = sum(P_avg_dilute.values())
    print(f"T = {temp}: sum of averaged P(n) = {total:.6f}")

    # Prepare data for plotting
    sizes_d = np.array(list(P_avg_dilute.keys()))
    probs_d = np.array(list(P_avg_dilute.values()))

    # Dense phase: report cluster sizes, not a distribution
    dense_sizes = [list(d.keys())[0] for d in dense_phase_CSD]
    dense_mean = np.mean(dense_sizes)
    dense_std = np.std(dense_sizes)

    dmax=n_tol_dict[temp]
    mask = (sizes_d >= 1) & (sizes_d <= dmax)    

    if(sizes_d[mask].shape[0] > 2):
        # Linear fit with covariance matrix
        coeffs, cov = np.polyfit(np.log10(sizes_d[mask]), np.log10(probs_d[mask]), 1, cov=True)
        slope_d, intercept_d = coeffs
        tau_d = -slope_d
        # Standard errors (sqrt of diagonal of covariance matrix)
        slope_err = np.sqrt(cov[0,0])
        tau_err = slope_err
        print(f"T = {temp}  τ = {tau_d:.3f} ± {tau_err:.3f}")
    else:
        slope_d, intercept_d = np.polyfit(np.log10(sizes_d[mask]), np.log10(probs_d[mask]), 1)
        tau_d = -slope_d
        tau_err = np.nan
        print(f"T = {temp}  τ = {tau_d:.3f}")

    # Fit curve for dilute
    P_fit_d = (10**intercept_d) * (sizes_d**slope_d)

    # Plot on first axis (1 ≤ n ≤ 100)
    temperature=float(temp)*5.6
    tau_rounded = float(np.round(tau_d,2))

    ax1.loglog(sizes_d, probs_d, marker=marker, linestyle='None', lw=1.0,markersize=5,color=color,alpha=0.7,label=r'$T = $'+f"{temperature:.2f}"+r'$\;\mathrm{K}$'+' , '+r'$\tau = $'+f"{tau_rounded:.2f}")
    #ax1.loglog(sizes_d, P_fit_d,linestyle='--',color='k',lw=0.7)
    
    temp_values.append(temp)
    tau_values.append(np.round(tau_d,4))
    if not np.isnan(tau_err):
        tau_error_values.append(np.round(tau_err,4))
    else:
        tau_error_values.append(np.nan)
    #secondlargestcluster_values.append(sizes_d[-1])
    largestcluster_values.append(np.median(sorted(dense_sizes)))
    
    #ax1.set_xlim(0, 1e5)
    #ax1.set_xticks([1, 10, 100, 1000, 10000, 100000])
    ax1.set_xlim(0, 1e2)
    ax1.set_xticks([1, 10, 100])
    ax1.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])

    #ax1.set_ylim(1e-5,10)
    ax1.legend(loc='upper right',fontsize=9)
    ax1.yaxis.tick_left()
    ax1.set_ylabel(r'$p(n)$')
    ax1.set_xlabel(r'$n$')

    #ax1.loglog(dense_sizes, [1.0]*len(dense_sizes), marker=marker, linestyle='None',lw=1.0, markersize=3, color='red', alpha=0.5)

plt.tight_layout()
plt.savefig(f'CSDplot_{variant}_L{boxsize}_Regime2_usingYFR.pdf',bbox_inches='tight')


T=["58.50","58.75","59","59.25"]

# --- store results ---
n_tol_dict = {}
slope_tol_dict = {}

for i,temp in enumerate(T):
    density=np.round((Nchains*chain_length)/(boxsize**3),4)
    pattern=f"./data_CSD_Nchains{Nchains}_Gaurav6e11_usingYFR/{variant}/CSDresults_T{temp}_L{boxsize}_frame*.npy"
    file_list=glob.glob(pattern)
    dense_phase_CSD=[]
    dilute_phase_CSD=[]
    for fp in file_list:
        if os.path.exists(fp):
            data = np.load(fp, allow_pickle=True)
            l=data.tolist()
            dense_phase_CSD.append(l['Dense_phase_CSD'])
            dilute_phase_CSD.append(l['Dilute_phase_CSD'])

    dense_phase_CSD=flatten_2d_list(dense_phase_CSD)
    dilute_phase_CSD=flatten_2d_list(dilute_phase_CSD)
    P_dense_seeds= dense_phase_CSD
    P_dilute_seeds= dilute_phase_CSD

    # Compute averaged distributions
    P_avg_dilute = average_distribution(P_dilute_seeds)

    # Prepare data for plotting
    sizes_d = np.array(list(P_avg_dilute.keys()))
    probs_d = np.array(list(P_avg_dilute.values()))

    dP_dn = compute_derivative(sizes_d, probs_d)

    dlogP_dlogn = compute_loglog_derivative(sizes_d, probs_d)
    curvature = np.gradient(dlogP_dlogn,np.log10(sizes_d))
    slope=dlogP_dlogn


# --- distinct markers for each T ---
markers = ['o', '^','s','p','*','X','D']  # reuse if more T

fig,ax1 = plt.subplots(figsize=(5.5, 3.5), dpi=300)

for i,temp in enumerate(T):

    marker = markers[i]
    color = color_dict[temp]

    density=np.round((Nchains*chain_length)/(boxsize**3),4)
    pattern=f"./data_CSD_Nchains{Nchains}_Gaurav6e11_usingYFR/{variant}/CSDresults_T{temp}_L{boxsize}_frame*.npy"
    file_list=glob.glob(pattern)
    dense_phase_CSD=[]
    dilute_phase_CSD=[]
    for fp in file_list:
        if os.path.exists(fp):
            data = np.load(fp, allow_pickle=True)
            l=data.tolist()
            dense_phase_CSD.append(l['Dense_phase_CSD'])
            dilute_phase_CSD.append(l['Dilute_phase_CSD'])

    dense_phase_CSD=flatten_2d_list(dense_phase_CSD)
    dilute_phase_CSD=flatten_2d_list(dilute_phase_CSD)
    P_dense_seeds= dense_phase_CSD
    P_dilute_seeds= dilute_phase_CSD

    # Compute averaged distributions
    P_avg_dilute = average_distribution(P_dilute_seeds)

    total = sum(P_avg_dilute.values())
    print(f"T = {temp}: sum of averaged P(n) = {total:.6f}")

    # Prepare data for plotting
    sizes_d = np.array(list(P_avg_dilute.keys()))
    probs_d = np.array(list(P_avg_dilute.values()))

    # Dense phase: report cluster sizes, not a distribution
    dense_sizes = [list(d.keys())[0] for d in dense_phase_CSD]
    dense_mean = np.mean(dense_sizes)
    dense_std = np.std(dense_sizes)

    # Plot on first axis (1 ≤ n ≤ 100)
    temperature=float(temp)*5.6

    #ax1.loglog(sizes_d, probs_d, marker=marker, linestyle='None', lw=1.0,markersize=5,color=color,alpha=0.7,label=r'$T = $'+f"{temperature:.2f}"+r'$\;\mathrm{K}$'+' , '+r'$\tau = $'+f"{tau_rounded:.2f}")
    ax1.loglog(sizes_d, probs_d, marker=marker, linestyle='None', lw=1.0,markersize=5,color=color,alpha=0.7,label=r'$T = $'+f"{temperature:.2f}"+r'$\;\mathrm{K}$')
    #ax1.loglog(sizes_d, P_fit_d,linestyle='--',color='k',lw=0.7)

    temp_values.append(temp)
    #secondlargestcluster_values.append(sizes_d[-1])
    largestcluster_values.append(np.median(sorted(dense_sizes)))

    #ax1.set_xlim(0, 1e5)
    #ax1.set_xticks([1, 10, 100, 1000, 10000, 100000])
    ax1.set_xlim(0, 1e2)
    ax1.set_xticks([1, 10, 100])
    ax1.set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])

    #ax1.set_ylim(1e-5,10)
    ax1.legend(loc='upper right',fontsize=9)
    ax1.yaxis.tick_left()
    ax1.set_ylabel(r'$p(n)$')
    ax1.set_xlabel(r'$n$')

    #ax1.loglog(dense_sizes, [1.0]*len(dense_sizes), marker=marker, linestyle='None',lw=1.0, markersize=3, color='red', alpha=0.5)

plt.tight_layout()
plt.savefig(f'CSDplot_{variant}_L{boxsize}_Regime3_usingYFR.pdf',bbox_inches='tight')


To=53.58
Tp=58.50
Tc=59.36
temperatures= np.array([float(t)*5.6 for t in temp_values])

np.save(f"tau_vs_T_data_{variant}_L{boxsize}_usingYFR.npy",np.array([temperatures[2:7],np.array(tau_values),np.array(tau_error_values)],dtype=object))
np.save(f"largestcluster_vs_T_data_{variant}_L{boxsize}_usingYFR.npy",np.array([temperatures,np.array(largestcluster_values)],dtype=object))
