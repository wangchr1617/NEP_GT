import h5py
import numpy as np
import os
import pandas as pd
import phonopy
import scienceplots
import seekpath
import subprocess
import sys
import warnings

from ase import Atoms
from ase.calculators.mixing import SumCalculator
from ase.io import read, write
from calorine.calculators import CPUNEP
from calorine.tools import relax_structure
from dftd3.ase import DFTD3
from hiphive import ForceConstantPotential
from matplotlib import pyplot as plt
from phonopy.api_gruneisen import PhonopyGruneisen
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import THzToCm

warnings.filterwarnings('ignore')
plt.style.use(['science', 'ieee', 'no-latex', 'bright'])

def calculate_primitive_matrix(unit_cell, primitive_cell):
    """
    计算原胞矩阵
    """
    return np.dot(primitive_cell, np.linalg.inv(unit_cell))

def scale_unit_cell(unit_cell, scale_factor):
    """通过给定的缩放因子缩放晶胞"""
    scaled_cell = unit_cell.copy()
    scaled_cell.set_cell(unit_cell.cell * scale_factor, scale_atoms=True)
    return scaled_cell

def nep_calculator(filename="./nep.txt", use_dftd3=False):
    """
    配置NEP计算器，若需色散校正，则use_dftd3=True
    """
    if os.path.exists(filename):
        if use_dftd3:
            return SumCalculator([CPUNEP(filename), DFTD3(method="pbe", damping="d3bj")])
        return CPUNEP(filename)
    print(f"File {filename} does not exist.")
    sys.exit(1)
        
def calculate_forces_by_nep(structure, nep="./nep.txt"):    
    """
    使用NEP计算结构的原子受力
    """
    structure.calc = nep_calculator(filename=nep)
    return structure.get_forces().copy()

def calculate_force_constants_by_dft(structure, primitive_matrix, supercell_matrix, unitcell_filename="POSCAR"):
    """
    加载并返回DFT计算的力常数
    """
    if os.path.exists("FORCE_SETS") or os.path.exists("FORCE_CONSTANTS"):
        phonon = phonopy.load(supercell_matrix=supercell_matrix, primitive_matrix=primitive_matrix, unitcell_filename=unitcell_filename)
        return phonon
    print("The file FORCE_SETS or FORCE_CONSTANTS does not exist.")
    return None

def calculate_force_constants_by_fcp(structure, primitive_matrix, supercell_matrix, fcp):    
    """
    使用hiphive计算力常数
    """
    phonopy_atoms = PhonopyAtoms(
        symbols=structure.symbols, 
        cell=structure.cell, 
        scaled_positions=structure.get_scaled_positions(), 
        pbc=[True, True, True]
    )
    phonon = phonopy.Phonopy(phonopy_atoms, supercell_matrix, primitive_matrix)
    phonopy_supercell = phonon.supercell
    structure_ase = Atoms(
        symbols=phonopy_supercell.symbols, 
        cell=phonopy_supercell.cell, 
        scaled_positions=phonopy_supercell.scaled_positions, 
        pbc=[True, True, True]
    )
    fcs = fcp.get_force_constants(structure_ase)
    phonon.force_constants = fcs.get_fc_array(order=2)
    return phonon

def calculate_force_constants_by_nep(structure, primitive_matrix, supercell_matrix, nep="./nep.txt", fmax=0.0001, distance=0.01):    
    """
    使用NEP计算力常数
    """
    structure.calc = nep_calculator(filename=nep)
    relax_structure(structure, fmax=fmax)
    phonopy_atoms = PhonopyAtoms(
        symbols=structure.symbols, 
        cell=structure.cell, 
        scaled_positions=structure.get_scaled_positions(), 
        pbc=[True, True, True]
    )
    phonon = phonopy.Phonopy(phonopy_atoms, supercell_matrix, primitive_matrix)
    phonon.generate_displacements(distance=distance)
    
    forces = []
    for phonopy_supercell in phonon.supercells_with_displacements:
        structure_ase = Atoms(
            symbols=phonopy_supercell.symbols, 
            cell=phonopy_supercell.cell, 
            scaled_positions=phonopy_supercell.scaled_positions, 
            pbc=[True, True, True]
        )
        forces.append(calculate_forces_by_nep(structure_ase, nep))
    
    phonon.forces = forces
    phonon.produce_force_constants()
    phonon.set_force_constants_zero_with_radius(cutoff_radius=20)
    phonon.save(settings={'force_constants': True})
    return phonon
    
def get_kpoints(structure):
    """
    生成k点路径
    """
    structure_tuple = (structure.cell, structure.get_scaled_positions(), structure.numbers)
    path = seekpath.get_explicit_k_path(structure_tuple)
    
    kpoints_rel, kpoints_lincoord, labels = path['explicit_kpoints_rel'], path['explicit_kpoints_linearcoord'], path['explicit_kpoints_labels']
    labels = ['$\Gamma$' if label == 'GAMMA' else label for label in labels]
    labels = [label.replace('_', '$_') + '$' if '_' in label else label for label in labels]
    
    return kpoints_rel, kpoints_lincoord, labels

def get_manual_kpoints(structure, high_symmetry_points, num=30):
    """
    基于手动定义的高对称点生成k点路径
    high_symmetry_points: 包含点坐标和标签的列表
    例如：[{'label': 'X', 'coords': [0.5, 0.0, 0.5]}, {'label': 'L', 'coords': [0.5, 0.5, 0.5]}]
    """
    kpoints_rel, kpoints_lincoord, labels = [], [], []
    current_distance = 0
    
    for i, point in enumerate(high_symmetry_points[:-1]):
        start, end = np.array(point['coords']), np.array(high_symmetry_points[i + 1]['coords'])
        kpoint_segment = np.linspace(start, end, num=num)
        kpoints_rel.extend(kpoint_segment)
        
        segment_length = np.linalg.norm(end - start)
        kpoints_lincoord.extend(np.linspace(current_distance, current_distance + segment_length, num=num))
        
        current_distance += segment_length
        labels.extend([point['label']] + [''] * (num-1))
        if i == len(high_symmetry_points) - 2:
            labels[-1] = high_symmetry_points[i + 1]['label']
        labels = ['$\Gamma$' if label == 'GAMMA' else label for label in labels]
    
    return kpoints_rel, kpoints_lincoord, labels

def create_phonon_dataframe(phonon, kpoints_rel, kpoints_lincoord):
    """
    生成声子数据的DataFrame
    """
    phonon.run_band_structure([kpoints_rel], with_eigenvectors=True, with_group_velocities=True)
    band_structure = phonon.get_band_structure_dict()
    
    df = pd.DataFrame(band_structure['frequencies'][0], index=kpoints_lincoord)
    return df
    
def calculate_dos(phonon):
    """
    计算声子态密度
    """
    phonon.run_mesh([20, 20, 20], is_mesh_symmetry=False, with_eigenvectors=False, with_group_velocities=False, is_gamma_center=False) 
    phonon.run_total_dos(sigma=0.05, freq_min=None, freq_max=None, freq_pitch=None, use_tetrahedron_method=False)
    dos = phonon.get_total_dos_dict()
    return dos

def calculate_msd(phonon, temperatures):
    """
    计算MSD
    """
    phonon.run_mesh([20, 20, 20], is_mesh_symmetry=False, with_eigenvectors=True, with_group_velocities=False, is_gamma_center=False) 
    phonon.run_thermal_displacements(temperatures=temperatures)
    thermal_displacements_dict = phonon.get_thermal_displacements_dict()
    msds = np.sum(thermal_displacements_dict["thermal_displacements"], axis=1)  # sum up the MSD over x,y,z
    return msds

def calculate_gruneisen_parameters(unit_cell, primitive_matrix, supercell_matrix, kpoints_rel, nep="./nep.txt", mesh_dims=[20, 20, 20], scale_factor=0.01):
    """
    计算格林艾森参数
    """
    unit_cell_plus = scale_unit_cell(unit_cell, 1 + scale_factor)
    write(f'plus_{scale_factor}.vasp', unit_cell_plus)
    unit_cell_minus = scale_unit_cell(unit_cell, 1 - scale_factor)
    write(f'minus_{scale_factor}.vasp', unit_cell_minus)
    
    phonon = calculate_force_constants_by_nep(unit_cell, primitive_matrix, supercell_matrix, nep)
    phonon_plus = calculate_force_constants_by_nep(unit_cell_plus, primitive_matrix, supercell_matrix, nep)
    phonon_minus = calculate_force_constants_by_nep(unit_cell_minus, primitive_matrix, supercell_matrix, nep)
    
    gruneisen = PhonopyGruneisen(phonon, phonon_plus, phonon_minus)
    gruneisen.set_mesh(mesh_dims)
    gruneisen.set_band_structure([np.array(kpoints_rel)])
    
    # band
    qpoints, distances_with_shift, frequencies, eigenvectors, gamma = gruneisen.get_band_structure()
    qpoints = np.array(qpoints).squeeze()
    distances_with_shift = np.array(distances_with_shift).squeeze()
    frequencies = np.array(frequencies).squeeze()
    eigenvectors = np.array(eigenvectors).squeeze()
    gamma = np.array(gamma).squeeze()
    n = len(gamma.T) - 1
    epsilon = 1e-4
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=300)
    
    for i, (curve, freqs) in enumerate(zip(gamma.T.copy(), frequencies.T)):
        if np.linalg.norm(qpoints[0]) < epsilon:
            cutoff_index = 0
            for j, q in enumerate(qpoints):
                if not np.linalg.norm(q) < epsilon:
                    cutoff_index = j
                    break
            for j in range(cutoff_index):
                if abs(freqs[j]) < abs(max(freqs)) / 10:
                    curve[j] = curve[cutoff_index]
        if np.linalg.norm(qpoints[-1]) < epsilon:
            cutoff_index = len(qpoints) - 1
            for j in reversed(range(len(qpoints))):
                q = qpoints[j]
                if not np.linalg.norm(q) < epsilon:
                    cutoff_index = j
                    break
            for j in reversed(range(len(qpoints))):
                if j == cutoff_index:
                    break
                if abs(freqs[j]) < abs(max(freqs)) / 10:
                    curve[j] = curve[cutoff_index]
        color = (1.0 / n * i, 0, 1.0 / n * (n - i))
        ax1.plot(distances_with_shift, curve, color=color, label=f'Mode {i+1}')
    ax1.set_xlim(0, distances_with_shift[-1])
    ax1.set_ylabel("$\gamma$")
    ax1.set_xticks([])
    ax1.legend(ncol=2, fontsize=plt.rcParams['font.size'] - 2)
    for i, freqs in enumerate(frequencies.T):
        color = (1.0 / n * i, 0, 1.0 / n * (n - i))
        ax2.plot(distances_with_shift, freqs, color=color, label=f'Mode {i+1}')
    ax2.set_xlim(0, distances_with_shift[-1])
    ax2.set_ylabel("Frequency (THz)")
    plt.tight_layout()
    plt.savefig("./gruneisen_band.png", bbox_inches='tight')
    plt.close()
    
    # mesh
    qpoints, weights, frequencies, eigenvectors, gamma = gruneisen.get_mesh()
    n = len(gamma.T) - 1
    plt.figure()
    for i, (g, freqs) in enumerate(zip(gamma.T, frequencies.T)):
        color = (1.0 / n * i, 0, 1.0 / n * (n - i))
        plt.plot(freqs, g, "o", color=color, markersize=1, label=f'Mode {i+1}')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("$\gamma$")
    plt.legend(ncol=2, fontsize=plt.rcParams['font.size'] - 2)
    plt.tight_layout()
    plt.savefig("./gruneisen_mesh.png", bbox_inches='tight')
    
def calculate_third_order_properties(structure, primitive_matrix, supercell_matrix, fcp, temp, filename, mesh=[20,20,20]):
    """
    计算声子寿命
    """
    phonopy_atoms = PhonopyAtoms(
        symbols=structure.symbols, 
        cell=structure.cell, 
        scaled_positions=structure.get_scaled_positions(), 
        pbc=[True, True, True]
    )
    phonon = phonopy.Phonopy(phonopy_atoms, supercell_matrix, primitive_matrix)
    phonopy_supercell = phonon.supercell
    structure_ase = Atoms(
        symbols=phonopy_supercell.symbols, 
        cell=phonopy_supercell.cell, 
        scaled_positions=phonopy_supercell.scaled_positions, 
        pbc=[True, True, True]
    )
    fcs = fcp.get_force_constants(structure_ase)
    fcs.write_to_phonopy('fc2.hdf5')
    fcs.write_to_phono3py('fc3.hdf5')
    
    h5py_filename = 'kappa-m{}{}{}.hdf5'.format(mesh[0], mesh[1], mesh[2])
    if not os.path.exists(h5py_filename):
        phono3py_cmd = 'phono3py --dim="{} {} {}" --fc2 --fc3 --br --mesh="{} {} {}" --ts="{}"'.format(
            supercell_matrix[0], supercell_matrix[1], supercell_matrix[2], 
            mesh[0], mesh[1], mesh[2], temp
        )
        print(phono3py_cmd)
        sys.exit(1)
    else:
        with h5py.File(h5py_filename, 'r') as hf:
            temperatures = hf['temperature'][:]
            frequency = hf['frequency'][:]
            gamma = hf['gamma'][:]
    
    plt.figure()
    dots = plt.plot(frequency.flatten(), gamma[0].flatten(), 'o', ms=1)
    
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Gamma (THz)')
    plt.xlim(0, None)
    plt.ylim(0, None)
    plt.legend((dots), ["{}".format(temp)], loc="best", fontsize=plt.rcParams['font.size'] - 2)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
           
def _plot_a_band(ax, df, color, label, **kwargs):
    """
    绘制一组声子色散曲线。
    """
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], color=color, label=label if i == 0 else None, **kwargs)
    
def _plot_a_dos(ax, phonon, color, label, **kwargs):
    """
    绘制一组声子色散曲线对应的态密度曲线。
    """   
    dos = calculate_dos(phonon)
    ax.plot(dos['frequency_points'], dos['total_dos'], color=color, label=label, **kwargs)

def _plot_band_and_dos(ax1, ax2, df, phonon, color, label, **kwargs):
    """
    绘制一组声子色散曲线及其对应的态密度曲线。
    """
    for i, col in enumerate(df.columns):
        ax1.plot(df.index, df[col], color=color, label=label if i == 0 else None, **kwargs)
        
    dos = calculate_dos(phonon)
    ax2.plot(dos['total_dos'], dos['frequency_points'], color=color, **kwargs)

def _plot_msd(ax, temperatures, msds, color, label, **kwargs):
    """
    绘制一条msd曲线
    """
    ax.plot(temperatures, msds, 'o-', color=color, label=label, **kwargs)
        
def plot_msd(temperatures, msds_dict, filename, **kwargs):
    """
    绘制msd曲线
    """
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (label, msds) in enumerate(msds_dict.items()):
        _plot_msd(ax, temperatures, msds, colors[i], label, **kwargs)
        
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('MSD ($Å^2$)')
    ax.legend(loc="best", fontsize=plt.rcParams['font.size'] - 2)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    
def plot_phonon_dispersion(kpoints_rel, kpoints_lincoord, labels, phonons, filename, **kwargs):
    """
    绘制声子色散。
    """
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (label, phonon) in enumerate(phonons.items()):
        df = create_phonon_dataframe(phonon, kpoints_rel, kpoints_lincoord)
        _plot_a_band(ax, df, colors[i], label, **kwargs)

    df_path = pd.DataFrame(dict(labels=labels, positions=kpoints_lincoord))
    df_path = df_path[df_path.labels != '']
    
    for xp in df_path.positions:
        ax.axvline(xp, c='black', lw=0.5)
    ax.axhline(y=0.0, c='black', lw=0.5)

    ax.set_xlabel('KPATH')
    ax.set_ylabel('Frequency (THz)')
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_xticks(df_path.positions)
    ax.set_xticklabels(df_path.labels)
    ax.legend(loc="upper right", fontsize=plt.rcParams['font.size'] - 2)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

def plot_phonon_dos(phonons, filename, **kwargs):    
    """
    绘制声子态密度图
    """
    fig, ax = plt.subplots()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (label, phonon) in enumerate(phonons.items()):
        _plot_a_dos(ax, phonon, colors[i], label, **kwargs)
    
    ax.set_xlabel('Frequency (THz)')
    ax.set_ylabel('DOS')
    ax.legend(loc="upper right", fontsize=plt.rcParams['font.size'] - 2)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

def plot_phonon_dispersion_and_dos(kpoints_rel, kpoints_lincoord, labels, phonons, filename, **kwargs):
    """
    绘制声子色散图（左边）和声子态密度图（右边），两个图共用一个 y 轴。
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [3, 1]})  # figsize=(7, 5)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (label, phonon) in enumerate(phonons.items()):
        df = create_phonon_dataframe(phonon, kpoints_rel, kpoints_lincoord)
        _plot_band_and_dos(ax1, ax2, df, phonon, colors[i], label, **kwargs)

    df_path = pd.DataFrame(dict(labels=labels, positions=kpoints_lincoord))
    df_path = df_path[df_path.labels != '']

    for xp in df_path.positions:
        ax1.axvline(xp, c='black', lw=0.5)
    ax1.axhline(y=0.0, c='black', lw=0.5)

    ax1.set_xlabel('KPATH')
    ax1.set_ylabel('Frequency (THz)')
    ax1.set_xlim(df.index.min(), df.index.max())
    ax1.set_xticks(df_path.positions)
    ax1.set_xticklabels(df_path.labels)
    ax1.legend(loc="upper right", fontsize=plt.rcParams['font.size'] - 2)

    ax2.axhline(y=0.0, c='black', lw=0.5)
    ax2.set_xlabel('DOS')
    ax2.set_ylabel(None)
    ax2.set_xlim(0, None)

    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
