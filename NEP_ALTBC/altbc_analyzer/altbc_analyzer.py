import math
import numpy as np
import pandas as pd
from .neighbor_list import NeighborList

class ALTBC_Analyzer:
    
    def __init__(self, cutoff=4.0, theta_min=155, theta_max=180):
        self.cutoff = cutoff
        self.theta_min = theta_min
        self.theta_max = theta_max

    def generate_neighbor_list(self, atoms):
        """
        Get neighbor list and atomic distances
        """
        nl = NeighborList(self.cutoff)
        return nl.find_neighbors(atoms)

    def find_triplets_with_distances(self, center_atom_index, neighbor_indices, distances, i_list, j_list):
        """
        Find all angle combinations and their corresponding atomic distances
        """
        triplets = []
        for first_neighbor in neighbor_indices:
            for second_neighbor in neighbor_indices:
                if first_neighbor != second_neighbor:
                    distance_1 = distances[(i_list == center_atom_index) & (j_list == first_neighbor)]
                    distance_2 = distances[(i_list == center_atom_index) & (j_list == second_neighbor)]
                    triplets.append([first_neighbor, center_atom_index, second_neighbor, distance_1[0], distance_2[0]])
        return np.array(triplets)

    def compute_altbc(self, atoms):
        """
        Calculate ALTBC
        """
        i_list, j_list, neighbor_distances = self.generate_neighbor_list(atoms)
        all_triplets, all_normalized_weights = [], []

        for center_atom_index in range(len(atoms)):
            neighbor_indices = j_list[i_list == center_atom_index]
            if len(neighbor_indices) > 1:
                triplets = self.find_triplets_with_distances(center_atom_index, neighbor_indices, neighbor_distances, i_list, j_list)
                all_triplets.extend(triplets)
                all_normalized_weights.extend([1.0 / (len(neighbor_indices))] * len(triplets))

        if not all_triplets:
            return pd.DataFrame()

        triplets = np.array(all_triplets)
        normalized_weights = np.array(all_normalized_weights).reshape(-1, 1)
        angles = atoms.get_angles(triplets[:, :3].astype(int), mic=True)
        
        df = pd.DataFrame(np.hstack([triplets, normalized_weights, angles[:, None]]), columns=["A", "B", "C", "AB", "BC", "weight", "angle_ABC"])
        df = df[(df["angle_ABC"] >= self.theta_min) & (df["angle_ABC"] <= self.theta_max)].reset_index(drop=True)
        df = df[(df["AB"] <= self.cutoff) & (df["BC"] <= self.cutoff)].reset_index(drop=True)
        df['pair'] = df.groupby(['A', 'B'])['weight'].transform('sum')    
        return df
