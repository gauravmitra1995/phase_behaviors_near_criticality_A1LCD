import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import LASSI_2 as lassi
import BinTrjProc as BR
import TrjProc as TP
import argparse
import matplotlib.colors as mcolors

def generate_random_subboxes(box_size, L, num_subboxes):
    np.random.seed(42) 
    print(str(num_subboxes)+" boxes are being generated....")
    x0 = np.random.uniform(0, box_size - L, size=num_subboxes)
    y0 = np.random.uniform(0, box_size - L, size=num_subboxes)
    z0 = np.random.uniform(0, box_size - L, size=num_subboxes)
    sub_box_df = pd.DataFrame({"x0": x0, "y0": y0, "z0": z0})
    return sub_box_df

def create_Chain_ID(coordinates, box_L, chain_size):
    grid = np.zeros((box_L, box_L, box_L), dtype=int)
    Chain_ID_grid = np.zeros((box_L, box_L, box_L), dtype=int)

    coordinates = np.array(coordinates, dtype=int)
     
    grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1

    chain_ids = np.arange(len(coordinates)) // chain_size + 1
    Chain_ID_grid[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = chain_ids

    return grid, Chain_ID_grid

def compute_subbox_density(grid, sub_box_df, L):
    densities = []
    for _, row in sub_box_df.iterrows():
        x0, y0, z0 = int(row['x0']), int(row['y0']), int(row['z0'])
        sub_box = grid[x0:x0+L, y0:y0+L, z0:z0+L]
        density = np.sum(sub_box) / (L ** 3)
        densities.append(density)

    sub_box_df['Density'] = densities
    return sub_box_df

parser = argparse.ArgumentParser()
parser.add_argument('--temp',type=str, required=True, help="Temp value.")
parser.add_argument('--boxsize',type=int,required=True, help="Box size.")
parser.add_argument('--N_subboxes',type=int,required=True, help="Number of subboxes.")
parser.add_argument('--variant',type=str,required=True,default='WTARO',help="Variant.")
parser.add_argument('--time_wa',type=int,required=True,help="Frame number.")

args = parser.parse_args()
temp = args.temp
boxsize = args.boxsize
N_subboxes =  args.N_subboxes
variant = args.variant
time_wa = args.time_wa


BOX_SIZE = boxsize
Nbeads=137
Nchains=10000
N_SUB_BOXES=N_subboxes
plot_number=200

print(f"Temp = {temp}, Boxsize = {boxsize}")
print(f"Starting all the computations for frame = {time_wa}......")

for L in [size for size in range(30, boxsize-20, 10)]: # Sub-box size
 
    print("===========================================================================================================================")
    print("Subdomain size L: ",L)

    subbox_densities_seeds=[]
    
    for seed in [1,2,3,4]:

        traj_path=f'/ceph/chpc/shared/rohit_v_pappu_group/gmitra/Percolation/{variant}/C{Nchains}_Restart5/L{boxsize}/T{temp}/{seed}/A1LCD_trj.lassi'

        if not os.path.exists(traj_path):
            print(f"Skipping seed {seed}: File does not exist.")
            continue  # Skip this seed and go to the next one

        Extractor_obj = BR.TrjExtractor(traj_path)
        BC = Extractor_obj.extract_coords()

        if len(BC) < 100:
            print(f"Skipping seed {seed}: Trajectory has only {len(BC)} frames (less than 100).")    
            continue  # Skip this trajectory and move to the next seed

        print(f"Processing seed {seed}, Shape of coordinate array: {BC.shape}")

        coordinates=BC[time_wa]
        random_subboxes = generate_random_subboxes(BOX_SIZE, L, N_SUB_BOXES)
        grid, _ = create_Chain_ID(coordinates, BOX_SIZE, Nbeads)
        subbox_densities = compute_subbox_density(grid, random_subboxes, L)
        subbox_densities_seeds.append(np.array(subbox_densities['Density'].tolist()))

    arr=np.array(subbox_densities_seeds)
    print(f"Shape of the array for L = {L} is: {arr.shape}")
    subbox_densities_seeds=arr
    np.save(f"./Density_subbox_Nchains{Nchains}_6e11_last100snaps/{variant}/Lbox{boxsize}/Density_T{temp}_L{L}_frame{time_wa}_Nsubboxes{N_SUB_BOXES}.npy",subbox_densities_seeds)

