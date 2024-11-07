import math
import numpy as np

class NeighborList:
    
    def __init__(self, cutoff):
        self.cutoff_neighbor = cutoff
        self.cutoff_square = cutoff ** 2

    def get_determinant(self, box):
        """
        Calculate the determinant of the box matrix
        """
        return (box[0] * (box[4] * box[8] - box[5] * box[7])
                + box[1] * (box[5] * box[6] - box[3] * box[8])
                + box[2] * (box[3] * box[7] - box[4] * box[6]))

    def get_area(self, a, b):
        """
        Calculate area
        """
        s1, s2, s3 = a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]
        return math.sqrt(s1 * s1 + s2 * s2 + s3 * s3)

    def get_thickness(self, box):
        """
        Calculate the thickness of the triclinic box
        """
        volume = abs(self.get_determinant(box))
        a, b, c = box[0:3], box[3:6], box[6:9]
        area_bc, area_ca, area_ab = self.get_area(b, c), self.get_area(c, a), self.get_area(a, b)
        return [volume / area_bc, volume / area_ca, volume / area_ab]

    def apply_minimum_image_convention_one(self, x12):
        """
        Apply minimum image convention to fractional coordinates
        """
        if x12 < -0.5:
            return x12 + 1.0
        elif x12 > 0.5:
            return x12 - 1.0
        return x12

    def apply_minimum_image_convention(self, box, x12, y12, z12):
        """
        Apply minimum image convention to a triclinic box
        """
        s = np.linalg.solve(box.reshape((3, 3)), np.array([x12, y12, z12]))
        s = np.array([self.apply_minimum_image_convention_one(s[i]) for i in range(3)])
        return np.dot(box.reshape((3, 3)), s)
    
    def find_cell(self, box, thickness, r, cutoff_inverse, num_cells):
        """
        Calculate the sub-box of an atom
        """
        s = np.linalg.solve(box.reshape((3, 3)), r)
        cell = [0, 0, 0, 0]
        for d in range(3):
            cell[d] = math.floor(s[d] * thickness[d] * cutoff_inverse)
            if cell[d] < 0:
                cell[d] += num_cells[d]
            if cell[d] >= num_cells[d]:
                cell[d] -= num_cells[d]
        cell[3] = cell[0] + num_cells[0] * (cell[1] + num_cells[1] * cell[2])
        return cell

    def find_neighbors(self, atoms, apply_mic=False):
        """
        Linear scaling algorithm for neighbor list with optional Minimum Image Convention (MIC)
        """        
        positions = atoms.get_positions()
        box = atoms.cell.array.flatten()
        thickness = self.get_thickness(box)
        cutoff_inverse = 1.0 / self.cutoff_neighbor
        
        num_cells = [max(1, math.floor(thickness[d] * cutoff_inverse)) for d in range(3)]
        num_cells.append(num_cells[0] * num_cells[1] * num_cells[2])
        
        cell_count = np.zeros(num_cells[3], dtype=int)
        cell_count_sum = np.zeros(num_cells[3], dtype=int)
        cell_contents = np.zeros(len(positions), dtype=int)
        i_list, j_list, d_list = [], [], []
    
        # First pass: Count atoms in each cell
        for n, r in enumerate(positions):
            cell = self.find_cell(box, thickness, r, cutoff_inverse, num_cells)
            cell_count[cell[3]] += 1
    
        # Compute cumulative sum of atom counts in each cell
        for i in range(1, num_cells[3]):
            cell_count_sum[i] = cell_count_sum[i - 1] + cell_count[i - 1]
    
        cell_count.fill(0)
    
        # Second pass: Assign atoms to cells
        for n, r in enumerate(positions):
            cell = self.find_cell(box, thickness, r, cutoff_inverse, num_cells)
            cell_contents[cell_count_sum[cell[3]] + cell_count[cell[3]]] = n
            cell_count[cell[3]] += 1
    
        # Neighbor search
        for n1, r1 in enumerate(positions):
            cell = self.find_cell(box, thickness, r1, cutoff_inverse, num_cells)
            for k, j, i in np.ndindex(3, 3, 3):
                neighbor_cell = cell[3] + ((k - 1) * num_cells[1] + (j - 1)) * num_cells[0] + (i - 1)
                if neighbor_cell < 0:
                    neighbor_cell += num_cells[3]
                if neighbor_cell >= num_cells[3]:
                    neighbor_cell -= num_cells[3]
                for m in range(cell_count[neighbor_cell]):
                    n2 = cell_contents[cell_count_sum[neighbor_cell] + m]
                    if n1 < n2:
                        x12, y12, z12 = positions[n2] - r1
                        # Apply Minimum Image Convention if apply_mic is True
                        if apply_mic:
                            x12, y12, z12 = self.apply_minimum_image_convention(box, x12, y12, z12)
                        d2 = x12 * x12 + y12 * y12 + z12 * z12
                        if d2 < self.cutoff_square:
                            i_list.append(n1)
                            j_list.append(n2)
                            d_list.append(math.sqrt(d2))
        
        return np.array(i_list), np.array(j_list), np.array(d_list)

