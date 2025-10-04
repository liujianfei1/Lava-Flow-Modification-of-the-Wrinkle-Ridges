import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import matplotlib
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

# 设置字体和样式
matplotlib.use('TkAgg')
sns.set_style('white')
plt.rcParams['font.family'] = 'Arial'

# 调整字号
plt.rcParams.update({
    'font.size': 16,  # 全局字体大小（轴标签、刻度、图例等）
    'font.weight': 'bold',  # 全局加粗
    'axes.labelsize': 18,  # x/y轴标签
    'xtick.labelsize': 18,  # x轴刻度标签
    'ytick.labelsize': 18,  # y轴刻度标签
    'axes.labelweight': 'bold',  # x/y轴标签加粗
    # 'axes.titleweight': 'bold',  # 标题加粗
    'legend.fontsize': 15,  # 图例文字大小
})

# 读取数据
data = pd.read_excel(r'E:\second_year\会议\文章\数据\Thickness_data_0527.xlsx')

# 拟合双程走时 vs 填充厚度
data_x = np.array(data[41:].fill_thickness)
data_y = np.array(data[41:].double_delay)
c = 300  # 光速
x = data_x[:, np.newaxis]
a, res, _, _ = np.linalg.lstsq(x, data_y, rcond=None)
residuals = data_y - (a * data_x)
SST = np.sum((data_y - data_y.mean())**2)
SSR = np.sum(residuals**2)
r2 = 1 - SSR / SST
xplot = np.linspace(0, 500, 100)

# 计算电常数和单层/双层厚度
epsilon = (a * c)**2 / 4
data['single_layer_thickness'] = c / np.sqrt(epsilon) * data.double_delay / 2
data['lower_layer_thickness'] = data.fill_thickness - data.single_layer_thickness

# 提取下层厚度（排除异常点）
layers = list(np.arange(0, 41))
a_index = np.delete(layers, [24, 27])
lower = [data.loc[i, 'lower_layer_thickness'] for i in a_index]

# 建图
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()  # 创建右侧 y 轴

# ========== 左轴：厚度-走时散点图 + 线性拟合 ========== #
ax1.plot(xplot, a * xplot, 'k--')

# 不同介电常数上下界线
emin, emax = 3, 20
kmin = 2 * np.sqrt(emin) / c
kmax = 2 * np.sqrt(emax) / c
ax1.plot(xplot, kmin * xplot, 'k-', linewidth=0.2)
ax1.plot(xplot, kmax * xplot, 'k-', linewidth=0.2)
ax1.fill_between(xplot, kmin * xplot, kmax * xplot, alpha=0.1, color='k')

# 拟合注释
ax1.annotate("y = {:.4f} x ".format(a[0]), [95, 1.64], rotation=70)
ax1.annotate("$r^2$ = {:.2f}".format(r2), [112, 1.7], rotation=70)

# 散点图及误差条
ax1.errorbar(data[41:].fill_thickness, data[41:].double_delay,
    xerr=data[41:].elevation_error_actual, fmt='o', elinewidth=0.5,
    label='CTX DEM (N=13)', alpha=0.5)
ax1.errorbar(data[0:41].fill_thickness, data[0:41].double_delay,
    xerr=data[0:41].elevation_error_actual, fmt='o', elinewidth=0.5,
    label='MOLA DEM (N=41)', alpha=0.5)

ax1.set_xlim(0, 500)
ax1.set_ylim(0., 2.2)
ax1.set_ylabel('Double travel time Δt (μs)')
ax1.set_xlabel('Fill thickness (m)')

# ========== 右轴：厚度分布直方图 + KDE ========== #
bins = np.arange(0, 400, 10)
sns.histplot(data.single_layer_thickness, bins=bins, stat='count', kde=True,
    kde_kws={'cut': 3}, label='L1 thickness (N={:d})'.format(len(data)), ax=ax2)
sns.histplot(lower, bins=bins, stat='count', kde=True, kde_kws={'cut': 3},
    label='L2 thickness (N={:d})'.format(len(lower)), color='orange', ax=ax2)

# KDE 曲线（总厚度）+ 找峰值
kde3 = gaussian_kde(data.fill_thickness, bw_method=0.4)
x3 = np.linspace(0, max(data.fill_thickness) * 1.1, 100)
kde3_vals = kde3(x3)
peaks, _ = find_peaks(kde3_vals)
print('总厚度的峰值为:', [(round(x3[p], 2), round(kde3_vals[p], 4)) for p in peaks])

# 保持乘数不变，保持曲线实际高度，但右轴压缩
ax2.plot(x3, kde3_vals * 550, color='black', label='Fill thickness (N={:d})'.format(len(data)), alpha=0.8)

# 压缩右轴纵向高度：放大右轴上限
ax2.set_ylim(0, 60)

# 右轴设置
ax2.set_ylabel('Crater count')
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=1, frameon=False, alignment='right')

# 布局与展示
plt.tight_layout()
plt.show()

plt.savefig(r'E:\second_year\会议\文章\图片\final\6厚度与介电常数\厚度与介电常数合并图v3.svg', dpi=300)  # 保存
plt.savefig(r'E:\second_year\会议\文章\图片\final\6厚度与介电常数\厚度与介电常数合并图v3.png', dpi=300)  # 保存
