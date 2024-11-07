from ase.io import read
from ase.io.vasp import read_vasp_xdatcar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import time
from scipy.stats import gaussian_kde
from altbc_analyzer.altbc_analyzer import ALTBC_Analyzer

def analyze_file(filename, filetype='POSCAR', cutoff=4.0, theta_min=155, theta_max=180, frame_interval=10):
    """
    Analyze input file type
    """
    analyzer = ALTBC_Analyzer(cutoff, theta_min, theta_max)
    
    if filetype == 'POSCAR':
        atoms = read(filename)
        return analyzer.compute_altbc(atoms)
    elif filetype == 'XDATCAR':
        structures = read_vasp_xdatcar(filename, index=slice(0, None, frame_interval))
        if len(structures[0]) > 1000:
            print("Please use POSCAR file type for analysis if the number of atoms exceeds 1000!")
            sys.exit(1)
        data_frames = [analyzer.compute_altbc(structure) for structure in structures]
        return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()
    else:
        print("Unrecognized file type!")
        return pd.DataFrame()

def plot_common_settings():
    plt.xlabel("Distance r1 (Å)")
    plt.ylabel("Distance r2 (Å)")
    plt.xlim(2.5, 3.8)
    plt.ylim(2.5, 3.8)
    plt.xticks(np.arange(2.6, 3.9, 0.2))
    plt.yticks(np.arange(2.6, 3.9, 0.2))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

def plot_results(df, plot_type='scatter'):
    if df.empty:
        print("No data available for plotting.")
        return
    
    x, y, z = np.array(df["AB"]), np.array(df["BC"]), np.array(df["pair"])
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    z_kde = kde(xy) * z * z  # Calculate z_kde here to use in both plots

    if plot_type == 'scatter':
        plot_scatter(x, y, z_kde)
    elif plot_type == 'kde':
        plot_kde(kde, x, y, z_kde)

def plot_scatter(x, y, z_kde):
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(x, y, c=z_kde, cmap=cmap, s=1, alpha=1)
    plt.colorbar(scatter, label='P(r1,r2)')
    plot_common_settings()

def plot_kde(kde, x, y, z_kde):
    x1, y1 = np.arange(2.5, 3.8, 0.01), np.arange(2.5, 3.8, 0.01)
    X, Y = np.meshgrid(x1, y1)
    kde_values = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    # Normalize kde_values to match the range of z_kde
    kde_values = kde_values * (z_kde.max() / kde_values.max())
    
    plt.figure(figsize=(6, 5))
    plt.imshow(kde_values, extent=[2.5, 3.8, 2.5, 3.8], origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='P(r1,r2)')
    plot_common_settings()

def main():
    start_time = time.time()
    
    try:
        # df = analyze_file('./XDATCAR', filetype='XDATCAR', frame_interval=10)
        df = analyze_file('./example/POSCAR', filetype='POSCAR')
        if not df.empty:
            plot_results(df, plot_type='scatter')  # 'scatter' or 'kde'
            plt.savefig('./altbc.png', bbox_inches='tight')
        else:
            print("No valid data found for plotting.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        
    end_time = time.time()
    print("Time used: {:.3f} s".format(end_time - start_time))

if __name__ == "__main__":
    main()
