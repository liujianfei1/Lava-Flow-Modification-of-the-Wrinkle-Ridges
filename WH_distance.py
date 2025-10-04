# 旧的更全的绘图版本在丢弃-皱脊横切面绘制.py中（包括最低点的位置；索引画图）
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import math
from sklearn.metrics import r2_score

matplotlib.use('TkAgg')
# 调整字号
plt.rcParams.update({
    'font.size': 16,  # 全局字体大小（轴标签、刻度、图例等）
    'axes.titlesize': 20,  # 坐标轴标题
    # 'axes.labelsize': 16,  # x/y轴标签
    'xtick.labelsize': 14,  # x轴刻度标签
    'ytick.labelsize': 14,  # y轴刻度标签
    'legend.fontsize': 15,  # 图例文字大小
})


def safe_legend(ax=None):
    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if any(not label.startswith('_') for label in labels):
        ax.legend()


def cal_distance(lon1, lat1, lon2, lat2):
    distance = math.sqrt((float(lon1) - lon2)**2 + (float(lat1) - lat2)**2)
    return distance


def ax_fit(x_data, y_data, cmt_x='', cmt_y='', cmt_title='', is_save=False, path_save='plot.png', use_sci=False, drawx=False, ax=None):
    # wr_width_manual, wr_h_avg_manual, 'Ridge width (km)', 'Ridge height (m)', 'Manual ridge height vs manual width', r"E:\second_year\交流记录\250624\人工高度随宽度变化.png"
    """
       绘图以及拟合函数，可显示 R² 并自动保存图片。
       参数：
           x_data, y_data : 数组，拟合输入数据
           cmt_x, cmt_y   : 字符串，x轴/y轴名
           cmt_title      : 字符串，标题名
           is_save        : 是否保存图片（布尔）
           path_save      : 保存路径
           ax : 是否要进行子图绘制
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.get_figure()

    # 线性拟合
    coef = np.polyfit(x_data, y_data, deg=1)
    fit_line = np.polyval(coef, x_data)
    sort_idx = np.argsort(x_data)
    x_data_sorted = x_data[sort_idx]
    y_data_sorted = y_data[sort_idx]
    fit_line_sorted = np.polyval(coef, x_data_sorted)

    # 计算 R²
    r2 = r2_score(y_data, fit_line)

    ax.scatter(x_data_sorted, y_data_sorted, color='k', s=10)
    ax.plot(x_data_sorted, fit_line_sorted, color='red', linestyle='--')  # , label='Fit line'

    # 调整为科学计数法坐标
    if use_sci:
        ax = plt.gca()
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 在图中标注拟合公式和 R²
    x_text = np.min(x_data) + 0.9 * (np.max(x_data) - np.min(x_data))
    y_text = np.min(fit_line) + 0.8 * (np.max(fit_line) - np.min(fit_line))
    ax.text(x_text, y_text, f'$R^2 = {r2:.2f}$', color='red', fontsize=12)

    # 字母标注
    # y_range = np.max(y_data_sorted) - np.min(y_data_sorted)
    # y_offset = 0.02 * y_range
    # for i, label in enumerate(y_data_sorted):
    #     ax.text(x_data_sorted[i], y_data_sorted[i] + y_offset, chr(ord('a') + i), ha='center', fontsize=9)

    if not drawx:
        # ax.set_xticks([])  # 不显示刻度线
        ax.set_xticklabels([])  # 不显示刻度值
        ax.set_xlabel('')  # 清除标签文本
    else:
        ax.set_xlabel(cmt_x)
    ax.set_ylabel(cmt_y)

    # ax.set_title(cmt_title, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    if any(not lbl.startswith('_') for lbl in labels):
        ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if is_save:
        fig.savefig(path_save, dpi=500)
    elif ax is None:
        plt.show()


def plot_fit(x_data, y_data, cmt_x='', cmt_y='', cmt_title='', is_save=False, path_save=r'D:\NEW_way\Desktop\皱脊横切面绘制.png', use_sci=False):  # wr_width_manual, wr_h_avg_manual, 'Ridge width (km)', 'Ridge height (m)', 'Manual ridge height vs manual width', r"E:\second_year\交流记录\250624\人工高度随宽度变化.png"
    """
       绘图以及拟合函数，可显示 R² 并自动保存图片。
       参数：
           x_data, y_data : 数组，拟合输入数据
           cmt_x, cmt_y   : 字符串，x轴/y轴名
           cmt_title      : 字符串，标题名
           is_save        : 是否保存图片（布尔）
           path_save      : 保存路径
           annotate_slope : 是否用倾斜角度标注 R²（默认 True）
    """
    coef = np.polyfit(x_data, y_data, deg=1)
    fit_line = np.polyval(coef, x_data)
    sort_idx = np.argsort(x_data)
    x_data_sorted = x_data[sort_idx]
    y_data_sorted = y_data[sort_idx]
    fit_line_sorted = np.polyval(coef, x_data_sorted)

    # 计算 R²
    r2 = r2_score(y_data, fit_line)

    plt.figure(figsize=(8, 5))

    plt.scatter(x_data_sorted, y_data_sorted, color='k', s=10)
    plt.plot(x_data_sorted, fit_line_sorted, color='red', linestyle='--')  # , label='Fit line'

    # 调整为科学计数法坐标
    if use_sci:
        ax = plt.gca()
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 在图中标注拟合公式和 R²
    x_text = np.min(x_data) + 0.9 * (np.max(x_data) - np.min(x_data))
    y_text = np.min(fit_line) + 0.8 * (np.max(fit_line) - np.min(fit_line))
    plt.text(x_text, y_text, f'$R^2 = {r2:.2f}$', color='red', fontsize=12)

    # 字母标注
    y_range = np.max(y_data_sorted) - np.min(y_data_sorted)
    y_offset = 0.02 * y_range
    for i, label in enumerate(y_data_sorted):
        plt.text(x_data_sorted[i], y_data_sorted[i] + y_offset, chr(ord('a') + i), ha='center', fontsize=9)

    plt.xlabel(cmt_x, fontsize=18)
    plt.ylabel(cmt_y, fontsize=18)
    # plt.title(cmt_title, fontsize=14)
    safe_legend()
    plt.grid(True)
    plt.tight_layout()
    if is_save:
        plt.savefig(path_save, dpi=500)


# 路径设置
path = r"F:\亚马逊平原皱脊\总数据"
path_w = os.path.join(path, "皱脊高度.csv")

# 读取 CSV 文件
df = pd.read_csv(path_w)
crater_x, crater_y = 2724.63664, 1102.683921

# 按照 x, y 分组，并提取每组的 x1, y1, RASTERVALU 列为数组
grouped = df.groupby(['x', 'y']).apply(
    lambda g: pd.Series({
        'x1_array': g['x1'].values,
        'y1_array': g['y1'].values,
        'RASTERVALU_array': g['RASTERVALU'].values
    })
).reset_index()

top_manual, toe_manual_w, toe_manual_e = np.array([21, 18, 20, 16, 20, 18, 18, 23, 22, 20, 19, 20, 20, 17, 18, 20, 20, 21]), np.array([10, 4, 18, 0, 15, 0, 1, 4, 14, 7, 7, 2, 10, 0, 2, 16, 18, 20]), np.array([37, 36, 22, 23, 24, 33, 25, 40, 37, 39, 27, 33, 31, 24, 23, 31, 22, 23])
wr_width_arcgis = np.array([11.5, 16.1, 4.1, 31, 6.5, 18.5, 13.4, 16, 11.7, 12.3, 4.99, 15.7, 11.6, 28.4, 5.8, 7.4, 6.75, 2.78])

wr_width_manual, wr_h_avg_manual, wr_h_east_manual, wr_h_west_manual, distance, wr_width_auto, wr_h_avg_auto, wr_h_east_auto, wr_h_west_auto = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

# 设置 4×5 子图
fig, axes = plt.subplots(4, 5, figsize=(30, 16))
axes_flat = axes.flatten()

for idx, row in grouped.iterrows():
    ax = axes_flat[idx]

    # 提取并排序
    x1, y1, elev = row['x1_array'][1:], row['y1_array'][1:], row['RASTERVALU_array'][1:]
    sort_idx = x1.argsort()
    x1_sorted, y1_sorted, elev_sorted = x1[sort_idx], y1[sort_idx], elev[sort_idx]

    # 人工拾取点
    top_idx, toe_w, toe_e, mid_idx = top_manual[idx], toe_manual_w[idx], toe_manual_e[idx], len(x1_sorted) // 2

    wr_width_manual = np.append(wr_width_manual, cal_distance(x1_sorted[toe_w], y1_sorted[toe_w], x1_sorted[toe_e], y1_sorted[toe_e]))
    h_w, h_e = elev_sorted[top_idx] - elev_sorted[toe_w], elev_sorted[top_idx] - elev_sorted[toe_e]
    wr_h_east_manual = np.append(wr_h_east_manual, h_e)
    wr_h_west_manual = np.append(wr_h_west_manual, h_w)
    wr_h_avg_manual = np.append(wr_h_avg_manual, (h_w + h_e) / 2)
    distance = np.append(distance, cal_distance(x1[mid_idx], y1[mid_idx], crater_x, crater_y))

    # 最大值和最小值自动得到高度
    minele_idx_e = np.argmin(elev_sorted[:mid_idx + 1])
    minele_idx_w = np.argmin(elev_sorted[mid_idx:]) + mid_idx
    wr_width_auto = np.append(wr_width_auto, cal_distance(x1_sorted[minele_idx_e], y1_sorted[minele_idx_e], x1_sorted[minele_idx_w], y1_sorted[minele_idx_w]))
    wr_h_east_auto = np.append(wr_h_east_auto, elev_sorted[top_idx] - elev_sorted[minele_idx_e])
    wr_h_west_auto = np.append(wr_h_west_auto, elev_sorted[top_idx] - elev_sorted[minele_idx_w])
    wr_h_avg_auto = np.append(wr_h_avg_auto, (elev_sorted[top_idx] - elev_sorted[minele_idx_w] + elev_sorted[top_idx] - elev_sorted[minele_idx_e]) / 2)

    # 画剖面曲线，人工拾取的点
    ax.plot(x1_sorted, elev_sorted, color='k')
    ax.scatter(x1_sorted[top_idx], elev_sorted[top_idx], color='red', s=20, label='Crest')
    ax.scatter(x1_sorted[toe_w], elev_sorted[toe_w], color='blue', s=20, label='Toe')
    ax.scatter(x1_sorted[toe_e], elev_sorted[toe_e], color='blue', s=20)

    # 标题和坐标
    ax.set_title(f'Ridge {chr(97 + idx)}', fontsize=12)
    ax.set_xlabel('Longitude (km)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.grid(True)
    # 每个子图只显示一次图例
    if idx == 0:
        ax.legend(loc='best', fontsize=8)

# 隐藏多余的空白子图
for j in range(len(grouped), len(axes_flat)):
    axes_flat[j].axis('off')

plt.tight_layout()
# plt.savefig(r'E:\second_year\会议\文章\图片\18条皱脊图.png', dpi=500)
#
# # 画图及拟合直线
# # 高度（两边的平均值，及两边）与宽度的图
# plot_fit(wr_width_manual, wr_h_avg_manual, 'Ridge width (km)', 'Ridge height (m)', 'Ridge height vs width', True, r"E:\second_year\会议\文章\图片\皱脊高度随宽度变化.png")
# # plot_fit(wr_width_auto, wr_h_avg_auto, 'Ridge width (km)', 'Ridge height (m)', 'Ridge height vs width', True, r"E:\second_year\会议\文章\图片\皱脊高度随宽度变化.png") # 自动
# # plot_fit(wr_width_manual, wr_h_west_manual, 'Ridge width (km)', 'Ridge western height (m)', 'Ridge western height vs width', True, r"E:\second_year\会议\文章\图片\皱脊西侧高度随宽度变化.png")
# # plot_fit(wr_width_manual, wr_h_east_manual, 'Ridge width (km)', 'Ridge eastern height (m)', 'Ridge eastern height vs width', True, r"E:\second_year\会议\文章\图片\皱脊东侧高度随宽度变化.png")
#
# # 宽度随距离的变化
# plot_fit(distance, wr_width_manual, 'Distance from Olympus Mons (km)', 'Ridge width (km)', 'Ridge width varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\皱脊宽度随距火山口距离变化.png")
#
# plot_fit(distance, (toe_manual_e - toe_manual_w) * 1.5, 'Distance from Olympus Mons (km)', 'Ridge width (km)', 'Ridge width varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\皱脊宽度_通过距离计算_随距火山口距离变化.png")
#
# plot_fit(distance, wr_width_arcgis, 'Distance from Olympus Mons (km)', 'Ridge width (km)', 'Ridge width (from ArcGIS) varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\皱脊宽度arcgis随距火山口距离变化.png")
# # plot_fit(distance, wr_width_auto, 'Distance from Olympus Mons (km)', 'Ridge width (km)', 'Ridge width varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\皱脊宽度随距火山口距离变化.png") # _auto'
#
#
# # 高度随距离变化
# plot_fit(distance, wr_h_avg_manual, 'Distance from Olympus Mons (km)', 'Ridge height (m)', 'Ridge height varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\皱脊高度随距火山口距离变化.png")
# # plot_fit(distance, wr_h_avg_auto, 'Distance from Olympus Mons (km)', 'Ridge height (m)', 'Ridge height varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\皱脊高度随距火山口距离变化.png") # _auto
#
#
# # 东西高度差/宽度 与 距离的关系
# height_diff_manual = wr_h_east_manual - wr_h_west_manual
# plot_fit(distance, height_diff_manual, 'Distance from Olympus Mons (km)', 'East west height difference (m)', 'East west height difference varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\东西高度差随距火山口距离变化.png")
#
# # 侵蚀高度与距离的关系
# num_erosion = 70 * 0.001  # 70
# height_org = wr_width_manual / num_erosion
# height_erosion = height_org - wr_h_avg_manual
#
# plot_fit(distance, height_erosion, 'Distance from Olympus Mons (km)', 'Erosion height (m)', 'Erosion height varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\侵蚀高度随距火山口距离变化.png")
#
# plot_fit(distance, wr_width_manual / wr_h_avg_manual * 1000, 'Width height ratio from Olympus Mons (km)', 'Width height ratio', 'Width height ratio varies with distance from Olympus Mons', True, r"E:\second_year\会议\文章\图片\宽高比随距火山口距离变化.png", True)
#
# # 还需要再看一下百分比的关系
#
# plt.close('all')
#
# plt.figure(figsize=(8, 5))
# plt.scatter(distance, wr_width_arcgis, color='b', label='arcgis', s=10)
# plt.scatter(distance, wr_width_manual, color='r', label='cross manual', s=10)
# # 字母标注
# for i, label in enumerate(wr_width_arcgis):
#     plt.text(distance[i], wr_width_arcgis[i] + 1, chr(ord('a') + i), ha='center', fontsize=9)
#     plt.text(distance[i], wr_width_manual[i] + 1, chr(ord('a') + i), ha='center', fontsize=9)
#
# plt.legend()

# 画三张子图
plt.figure(figsize=(8, 12))

plt.subplot(3, 1, 1)
ax_fit(distance, (toe_manual_e - toe_manual_w) * 1.5,
    'Distance from Olympus Mons (km)', 'Ridge width (km)',
    'Ridge width varies with distance from Olympus Mons',
    False, ax=plt.gca())

plt.subplot(3, 1, 2)
ax_fit(distance, wr_h_avg_manual,
    'Distance from Olympus Mons (km)', 'Ridge height (m)',
    'Ridge height varies with distance from Olympus Mons',
    False, ax=plt.gca())

plt.subplot(3, 1, 3)
ax_fit(distance, (toe_manual_e - toe_manual_w) * 1.5 / wr_h_avg_manual * 1000,
    'Distance from Olympus Mons (km)', 'Width-to-height ratio',
    'Width height ratio varies with distance from Olympus Mons',
    False, use_sci=True, drawx=True, ax=plt.gca())

plt.tight_layout()
plt.savefig(r"E:\second_year\会议\文章\图片\final\皱脊高度-宽度-宽高比\三图随距火山口距离变化v2.svg", dpi=300)

# 输出
whr = []
for i in range(len(distance)):
    width = (toe_manual_e[i] - toe_manual_w[i]) * 1.5
    whr.append(width / wr_h_avg_manual[i] * 1000)
    print(f"{width:.1f}\t{wr_h_west_manual[i]:.1f}\t{wr_h_east_manual[i]:.1f}\t{wr_h_avg_manual[i]:.1f}\t{width / wr_h_avg_manual[i] * 1000:.0f}\t{distance[i]:.0f}")
