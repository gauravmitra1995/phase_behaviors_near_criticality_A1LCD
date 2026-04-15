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

boxsize=120
Nchains=1
chain_length=137 
variant='WTARO'

temps=[35.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0]

overlap_T=[]
overlap_T_error=[]
Ree_T=[]

for temp in temps:
    
    Ree_rms_allseeds=[]
    overlap_allseeds=[]

    for seed in [1,2,3]:    
        Extractor_obj=BR.TrjExtractor('/project/fava/work/minafarag/A1LCD/Spacer_Production_NQT_2_Single/No_NCPR_Term/E0/'+str(variant)+'/R1/T'+str(temp)+'/B'+str(boxsize)+'/P'+str(boxsize)+'/C'+str(Nchains)+'/'+str(seed)+'/A1LCD_trj.lassi')
        BC=Extractor_obj.extract_coords()
        
        Ree_list = []
        for frame in range(BC.shape[0]-20000,BC.shape[0],1):
        
            coord_list=BC[frame]
    
            r_end=coord_list[-1].astype(float)
            r_start=coord_list[0].astype(float)
    
            # Apply minimum image convention
            delta = r_end - r_start
            delta -= boxsize * np.round(delta / boxsize)  # wrap distances

            # Compute PBC-corrected end-to-end distance
            Ree = np.linalg.norm(delta)
            Ree_list.append(Ree)
        
        # Compute RMS end-to-end distance
        
        Ree_rms = np.sqrt(np.mean(np.square(Ree_list)))
        Ree_rms_allseeds.append(Ree_rms)
        overlap_allseeds.append((chain_length*(0.5**3))/((Ree_rms)**3))
    
    Ree_rms_allseeds=np.array(Ree_rms_allseeds)  
    overlap_allseeds=np.array(overlap_allseeds)
    
    Ree_T.append(np.mean(Ree_rms_allseeds))
    overlap_T.append(np.mean(overlap_allseeds))
    overlap_T_error.append(np.std(overlap_allseeds))
    
Ree_T=np.array(Ree_T)
overlap_T=np.array(overlap_T)
overlap_T_error=np.array(overlap_T_error)

from scipy.interpolate import interp1d

# Interpolator (cubic spline)
phi_interp = interp1d(temps, overlap_T, kind='cubic')

# Evaluate at intermediate temperatures
temps_interp = np.linspace(35, 70, 1000)  # Finer resolution
overlap_interp = phi_interp(temps_interp)

# Plot
plt.figure(figsize=(6, 3.5), dpi=300)
plt.plot(temps_interp*5.6, overlap_interp, '--', lw=2.0,color='k',label='Interpolated')
plt.errorbar(np.array(temps)*5.6, overlap_T, yerr=overlap_T_error, ls='None',marker='o', markersize=10,capsize=10,elinewidth=4.0, color='crimson', label='Data')
plt.xlabel(r'$T\;(\mathrm{K})$')
plt.ylabel(r'$\phi^\ast(T)$')
plt.legend(loc='best')
plt.xticks(np.array([35,40,45,50,55,60,65,70])*5.6)
plt.tight_layout()
plt.savefig('phioverlap_vs_T_singlechain_WT_tempsinKelvin.pdf',bbox_inches='tight')

temps_of_interest=np.array([46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,64.0])
phi_overlap = np.array([phi_interp(T)for T in temps_of_interest])

#Plot
plt.figure(figsize=(6, 3.5), dpi=300)
plt.plot(temps_of_interest*5.6, phi_overlap, ls='None', marker='o',markersize=10,lw=2.0,color='crimson')
plt.xlabel(r'$T\;(\mathrm{K})$')
plt.ylabel(r'$\phi^\ast(T)$')
plt.xticks(np.array([45,50,55,60,65])*5.6)
plt.ylim(0.002,0.006)
plt.tight_layout()
plt.savefig('phioverlap_vs_T_singlechain_WT_zoomed_tempsinKelvin.pdf',bbox_inches='tight')

# Define desired interpolation temperatures
temps_of_interest = np.array([
    40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,
    50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,
    60.0,61.0,62.0,63.0,64.0,65.0
])

# Create output folder if it doesn't exist
output_folder = "overlap_variants"
os.makedirs(output_folder, exist_ok=True)

variants=['WTARO']

# Interpolate and save
for variant in variants:
    phi_interp = interp1d(temps, overlap[variant], kind='cubic')
    phi_overlap_interp = np.array([phi_interp(T) for T in temps_of_interest])

    df = pd.DataFrame({
        'Temperature': temps_of_interest,
        'phi_star_interp': phi_overlap_interp
    })

    output_path = os.path.join(output_folder, f'{variant}_phi_overlap_interpolated.csv')
    df.to_csv(output_path, index=False)


