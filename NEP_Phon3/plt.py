import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
from matplotlib.colors import LogNorm

# 打开 HDF5 文件
file_path = 'kappa-m202020.hdf5'
f = h5.File(file_path, 'r')

# 读取频率和线宽数据
frequencies = np.array(f['frequency'])  # 形状：(781, 6)
linewidths = np.array(f['gamma']).squeeze()  # 形状：(781, 6)
q_points = np.array(f['qpoint'])  # 形状：(781, 3)

# 处理线宽数据，避免 LogNorm 无法处理 0 的问题
epsilon = 1E-2
linewidths = np.maximum(linewidths, epsilon)

# 读取 x 轴刻度和标签
xt = np.array(f.get('q_ticks', []))  # 使用空列表作为默认值
q_tick_labels = f.attrs.get('q_tick_labels')

# 检查是否存在 q_tick_labels 属性
if q_tick_labels is not None:
    xl = q_tick_labels.split()
else:
    xl = []  # 或者你可以提供默认的标签

# 绘制频率色散曲线
plt.figure(figsize=(8, 6))
for i in range(frequencies.shape[1]):
    plt.plot(np.arange(frequencies.shape[0]), frequencies[:, i], color='b')

plt.xticks(xt, xl)
plt.xlabel('K-point Path')
plt.ylabel('Frequency (THz)')
plt.title('Phonon Band Structure Along High Symmetry Path')

# 保存频率色散曲线
plt.tight_layout()
plt.savefig('phonon_band_structure.png')

# 绘制线宽色散曲线
plt.figure(figsize=(8, 6))
plt.pcolormesh(np.arange(linewidths.shape[0]), np.arange(linewidths.shape[1]), linewidths.T, 
               norm=LogNorm(vmin=linewidths.min(), vmax=linewidths.max()), cmap='afmhot', shading='auto')

plt.xticks(xt, xl)
plt.xlabel('K-point Path')
plt.ylabel('Band Index')
plt.title('Phonon Linewidth Along High Symmetry Path')

# 保存线宽色散曲线
plt.tight_layout()
plt.savefig('phonon_linewidth.png')

# 关闭 HDF5 文件
f.close()


-------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

# 打开HDF5文件
f = h5.File('kappa-m202020.hdf5', 'r')

# 读取数据
frequencies = np.array(f.get('frequencies'))  # 假设频率数据的shape是(781, 6)
qpoints = np.array(f.get('qpoints'))  # 假设qpoints的shape是(781, 3)

# 高对称点路径
high_symmetry_points = np.array([
    [0.0, 0.0, 0.0],  # Γ
    [0.5, 0.5, 0.5],  # X
    [0.5, 0.0, 0.5],  # M
    [0.0, 0.0, 0.0]   # Γ (loop back)
])

# 计算路径长度
def path_length(qpoints, high_symmetry_points):
    lengths = [0.0]
    total_length = 0.0
    for i in range(1, len(high_symmetry_points)):
        segment_length = np.linalg.norm(high_symmetry_points[i] - high_symmetry_points[i-1])
        total_length += segment_length
        lengths.append(total_length)
    return lengths

# 获取路径长度
lengths = path_length(qpoints, high_symmetry_points)

# 绘制声子色散图
plt.figure(figsize=(10, 6))

for i in range(frequencies.shape[1]):
    plt.plot(lengths, frequencies[:, i], label=f'Mode {i+1}')

plt.xlabel('Path Length (Å)')
plt.ylabel('Frequency (THz)')
plt.title('Phonon Dispersion')
plt.legend()
plt.grid(True)
plt.savefig('phonon_dispersion.png')

