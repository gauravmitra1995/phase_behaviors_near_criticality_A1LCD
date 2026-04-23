import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import seaborn as sns

from skimage import measure
sns.set()
sns.set_style("white")
sns.set_style("ticks")
sns.set_context("poster")
import pickle

import os
import LASSI_2 as lassi
import BinTrjProc as BR
import TrjProc as TP
import argparse
from mpl_toolkits.mplot3d import Axes3D 
from scipy.spatial import cKDTree
from collections import defaultdict
from collections import deque
import pandas as pd

def flatten_2d_list(two_d_list):
    flattened_list = [item for sublist in two_d_list for item in sublist]
    return flattened_list

def build_chain_adjacency_list(chains, box_size):
    n_chains = len(chains)
    adjacency_list = {i: [] for i in range(n_chains)}

    bead_to_chain = {}
    for chain_idx, chain in enumerate(chains):
        for bead in chain:
            bead_tuple = tuple(np.array(bead) % box_size)
            bead_to_chain[bead_tuple] = chain_idx

    neighbors = [(dx, dy, dz)
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)]

    for bead, chain_idx in bead_to_chain.items():
        x, y, z = bead
        for dx, dy, dz in neighbors:
            neighbor_bead = tuple(np.array([x + dx, y + dy, z + dz]) % box_size)   # Apply PBC wrapping
            if neighbor_bead in bead_to_chain:
                neighbor_chain_idx = bead_to_chain[neighbor_bead]
                if chain_idx != neighbor_chain_idx:  # Only connect different chains
                    if neighbor_chain_idx not in adjacency_list[chain_idx]:
                        adjacency_list[chain_idx].append(neighbor_chain_idx)
                    if chain_idx not in adjacency_list[neighbor_chain_idx]:
                        adjacency_list[neighbor_chain_idx].append(chain_idx)

    return adjacency_list,bead_to_chain

def find_sticker_positions_single(sequence, letters):
    seq = sequence.replace("\n","").replace(" ","")
    return [i for i,ch in enumerate(seq) if ch in letters]

def build_chain_adjacency_list_stickers(chains,box_size,sequences,letters):
    n_chains = len(chains)
    adjacency_list = {i: [] for i in range(n_chains)}

    sticker_idxs=find_sticker_positions_single(sequences,letters)
    print("Sticker indexes are: ",sticker_idxs)
    print("We are measuring connectivities only using sticker beads now....")

    # 2) only insert sticker beads into the lookup
    bead_to_chain = {}
    for chain_idx, chain in enumerate(chains):
        for bead_idx, bead in enumerate(chain):
            # ← skip non‐stickers
            if bead_idx not in sticker_idxs:
                continue

            bead_tuple = tuple(np.array(bead) % box_size)
            bead_to_chain[bead_tuple] = chain_idx

    # 3) same 26‐neighbor offsets as before
    neighbors = [(dx, dy, dz)
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)]
    # 4) build adjacency exactly as you had it
    for bead, chain_idx in bead_to_chain.items():
        x, y, z = bead
        for dx, dy, dz in neighbors:
            neighbor_bead = tuple(np.array([x + dx, y + dy, z + dz]) % box_size)
            if neighbor_bead in bead_to_chain:
                neighbor_chain_idx = bead_to_chain[neighbor_bead]
                if chain_idx != neighbor_chain_idx:
                    if neighbor_chain_idx not in adjacency_list[chain_idx]:
                        adjacency_list[chain_idx].append(neighbor_chain_idx)
                    if chain_idx not in adjacency_list[neighbor_chain_idx]:
                        adjacency_list[neighbor_chain_idx].append(chain_idx)

    return adjacency_list,bead_to_chain

def find_largest_chain_cluster(adjacency_list):
    """
    Find the largest connected cluster of chains.
    """
    visited = set()
    largest_cluster = []
    max_cluster_size = 0

    def dfs(chain_idx, cluster):
        """ Perform DFS to find all chains in the same connected component. """
        stack = [chain_idx]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                cluster.append(current)
                stack.extend(adjacency_list[current])

    # Identify all clusters
    for chain_idx in adjacency_list:
        if chain_idx not in visited:
            cluster = []
            dfs(chain_idx, cluster)
            if len(cluster) > max_cluster_size:
                max_cluster_size = len(cluster)
                largest_cluster = cluster

    return largest_cluster, max_cluster_size


def extract_chains_from_largest_cluster(chains, largest_cluster_indices):
    return [chains[idx] for idx in largest_cluster_indices]

parser = argparse.ArgumentParser()
# Add arguments for temp
parser.add_argument('--temp',type=str, required=True, help="Temp value.")
parser.add_argument('--boxsize',type=int,required=True, help="Box size.")
parser.add_argument('--Nchains',type=int,required=True,default=10000, help="Number of chains.")
parser.add_argument('--variant',type=str,required=True,default='WTARO',help="Variant.")
parser.add_argument('--time_wa',type=int,required=True, help="Frame number.")
# Parse the arguments
args = parser.parse_args()
temp = args.temp
boxsize = args.boxsize
Nchains = args.Nchains
variant = args.variant
time_wa = args.time_wa

chain_length=137

print("=======================================================================================")
print("Temperature is:",temp)

density=np.round((Nchains*chain_length)/(boxsize**3),4)
print("Box size is: ",boxsize)

seq="GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF"
targets = ['Y','F','R']

largestclustersize_allseeds=[]
phic_allseeds=[]

for seed in [1,2,3,4]:

    traj_path=f'/ceph/chpc/shared/rohit_v_pappu_group/gmitra/Percolation/{variant}/C{Nchains}_Restart5/L{boxsize}/T{temp}/{seed}/A1LCD_trj.lassi'


    # Check if the file exists
    if not os.path.exists(traj_path):
        print(f"Skipping seed {seed}: File does not exist.")
        continue  # Skip this seed and go to the next one

    # Extract trajectory if file exists
    Extractor_obj = BR.TrjExtractor(traj_path)
    BC = Extractor_obj.extract_coords()

    if len(BC) < 100:
        print(f"Skipping seed {seed}: Trajectory has only {len(BC)} frames.")
        continue  # Skip this trajectory and move to the next seed

    print(f"Processing seed {seed}, Shape of coordinate array: {BC.shape}")

    frameno=time_wa
    print(f"But we are interested in frame {frameno}")

    tmp = np.arange(1,Nchains+1,1)
    coord_list=BC[frameno]
    particles_in_the_cluster=[coord_list[(ele-1)*chain_length:ele*chain_length].tolist() for ele in tmp]
    chains=particles_in_the_cluster
    chain_adj_list,bead_to_chain = build_chain_adjacency_list_stickers(chains, boxsize,seq,targets)
    #chain_adj_list,bead_to_chain = build_chain_adjacency_list(chains, boxsize)

    print(len(bead_to_chain.keys()))

    largest_chain_cluster, largest_cluster_size = find_largest_chain_cluster(chain_adj_list)
    print("Size of the Largest Chain Cluster:", largest_cluster_size)
    largest_cluster_chains = extract_chains_from_largest_cluster(chains, largest_chain_cluster)
 
    largestclustersize_allseeds.append(largest_cluster_size)

    phi_c=np.round(largest_cluster_size/Nchains,4)
    print("Fraction of chains in largest cluster is: ",phi_c)
    phic_allseeds.append(phi_c)

# Create a dictionary with your results
results = {
    "Temperature": temp,
    "System Density": density,
    "Box size": boxsize,
    "Largest Cluster Size": largestclustersize_allseeds,
    "phi_c": phic_allseeds
}

# Save the dictionary as an .npy file
np.save(f"./data_phic_Nchains10000_Gaurav6e11_usingYFR/{variant}/clusterresults_T{temp}_L{boxsize}_frame{time_wa}.npy", results)
