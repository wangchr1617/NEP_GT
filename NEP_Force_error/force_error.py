import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.rcParams['font.size'] = 14
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
import warnings
warnings.filterwarnings('ignore')

def plot(e_d, c, l):
    plt.hist(e_d, bins=30, color=c, alpha=0.3, density=True, label=l)
    xmin, xmax = plt.xlim()
    x_d = np.linspace(xmin, xmax, 100)
    mu_d, std_d = norm.fit(e_d)
    p_d = norm.pdf(x_d, mu_d, std_d)
    plt.plot(x_d, p_d, linewidth=1, linestyle='--', color=c, alpha=0.75)
    plt.fill_between(x_d, p_d, where=(p_d>=0), color=c, alpha=0.15)

def force_err():
    r = np.loadtxt('./ref.txt').reshape((-1,1))
    d = np.loadtxt('./dft.txt').reshape((-1,1))
    e = np.loadtxt('./encut.txt').reshape((-1,1))
    k = np.loadtxt('./kpoints.txt').reshape((-1,1))
    n = np.loadtxt('./nep.txt').reshape((-1,1))
    n = np.concatenate((r, d, e, k, n), axis=1)
    df = pd.DataFrame(n, columns=["ref","dft","encut=300","kpoints=1","nep"])
    df["e_d"] = np.array(df["ref"]) - np.array(df["dft"])
    df["e_e"] = np.array(df["ref"]) - np.array(df["encut=300"])
    df["e_k"] = np.array(df["ref"]) - np.array(df["kpoints=1"])
    df["e_n"] = np.array(df["ref"]) - np.array(df["nep"])
    e_d = np.array(df['e_d'])
    e_e = np.array(df['e_e'])
    e_k = np.array(df['e_k'])
    e_n = np.array(df['e_n'])   

    plt.figure(figsize=(5, 4), dpi=300)
    plot(e_k, c="skyblue", l="DFT")
    plot(e_n, c="red", l="NEP")

    plt.xlabel('Force error (eV/A)')
    plt.ylabel('Density')
    plt.xlim(-0.25, 0.25)
    plt.xticks(np.arange(-0.2, 0.25, 0.1))
    plt.ylim(0, 10)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./Force_error.png', bbox_inches='tight')

force_err()

