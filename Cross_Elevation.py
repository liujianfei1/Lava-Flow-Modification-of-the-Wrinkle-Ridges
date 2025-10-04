from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import os
import scipy.signal as sg
from matplotlib.lines import Line2D

matplotlib.use('TkAgg')
# 调整字号
plt.rcParams.update({
    'font.size': 13,  # 全局字体大小（轴标签、刻度、图例等）
    'axes.titlesize': 15,  # 坐标轴标题
    'xtick.labelsize': 12,  # x轴刻度标签
    'ytick.labelsize': 12,  # y轴刻度标签
    'legend.fontsize': 12,  # 图例文字大小
})


def get_minima(values):
    value_arr = np.array(values)
    min_index = sg.argrelmin(value_arr)[0]  # 获取极小值的下标
    return min_index, value_arr[min_index]  # 返回极小值的下标和对应的值


def draw_selected(wrs, path, space, selected_indices=[2, 3, 4, 8]):
    # 子图数 = 选中的皱脊数量
    num_row = len(selected_indices)
    fig, axes = plt.subplots(num_row, 1, figsize=(5, 8.5))
    if num_row == 1:
        axes = [axes]  # 保证 axes 可迭代

    for idx, ax, i in zip(range(num_row), axes, selected_indices):
        value = wrs[i]
        wr_mola = [row[1] for row in value]
        wr_sub = [row[2] for row in value]
        ttime = [(a - b) * math.sqrt(7.50) / 300 for a, b in zip(wr_mola, wr_sub)]
        wr_x = list(np.arange(0, len(wr_mola) * space, space))

        # 创建每个子图内的 2 个子图
        ax1 = ax.inset_axes([0, 0, 1, 0.8])  # 第一个子图的位置和大小
        ax2 = ax.inset_axes([0, 0.8, 1, 0.2])  # 第二个子图的位置和大小

        # ax1: surface & subsurface
        ax1.plot(wr_x, wr_mola, label='surface', color='tab:blue')
        ax1.plot(wr_x, wr_sub, label='subsurface', color='tab:orange')
        ax1.set_ylabel("H (m)")
        ax1.yaxis.set_major_locator(plt.MaxNLocator(2))
        # x 轴仅最后一个子图显示
        if idx != num_row - 1:
            ax1.set_xticks([])
            ax1.set_xlabel('')
        else:
            ax1.set_xlabel("Distance (km)")

        # ax2: travel time
        ax2.plot(wr_x, ttime, label='travel time', color='tab:green')
        ax2.set_ylabel("dt ($\mu$s)", rotation=0, labelpad=37, va='center')
        ax2.set_xticks([])

        # 序号标签
        labeltext = 'c' + str(idx + 1)
        ax2.annotate(labeltext, xy=(-0.15, 1.3), xycoords='axes fraction',
            fontweight='bold', ha='center', va='center')

        ax.axis('off')  # 隐藏父图坐标轴

    # 统一图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='tab:blue', label='surface'),
        Line2D([0], [0], color='tab:orange', label='subsurface'),
        Line2D([0], [0], color='tab:green', label='travel time')
    ]
    # fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(os.path.join(path, r'E:\second_year\会议\文章\图片\final\8皱脊地下高程\皱脊横切_克里金v5.svg'), dpi=300)
    plt.close()


def cal_w_h(data, num_left, num_right, space=0.5):
    wr_x = list(np.arange(0, len(data) * space, space))
    max_index = data.index(max(data))
    print("最大值的索引:", max_index * space, "极小值的索引:", get_minima(data)[0], get_minima(data)[0] * space)
    print("最大值:", max(data), "极小值:", get_minima(data)[1])
    # 计算坡度（一阶导数，离散化处理）
    slope = np.gradient(data, wr_x)
    # 设定一个阈值，检测坡度接近0的位置
    threshold = 0.1  # 根据数据适当调整这个阈值
    foot_indices = np.where(np.abs(slope) < threshold)[0]  # 找到坡度小于阈值的点
    print("坡脚:", foot_indices, foot_indices * space)

    width = (num_right - num_left) * space
    height_west = max(data) - data[num_left]
    height_east = max(data) - data[num_right]
    height = (height_west + height_east) / 2
    print("宽度,西侧高度，东侧高度，平均高度，高宽比分别为{:.1f} {:.1f} {:.1f} {:.1f} {:.2f}".format(width, height_west, height_east, height, height / width))

    # 缩短前的长度
    # length = 0
    # for i in range(num_left, num_right):
    #     length += math.sqrt((data[i] - data[i + 1])**2 + (space * 1000)**2)


def simpleplt(y, title='', label=['surface', 'subsurface']):
    if type(y) != list:
        y = [y]
    plt.figure(figsize=(8, 4))
    for n in range(len(y)):
        x = np.arange(len(y[n]))
        plt.plot(x, y[n], label=label[n])
    plt.xlabel('Index')
    plt.ylabel('Elevation (m)')
    plt.title('Elevation of cross ' + str(title))
    plt.grid(True)

    # 自动添加图例（如果需要）
    handles, labels = plt.gca().get_legend_handles_labels()
    if any(not lbl.startswith('_') for lbl in labels):
        plt.legend()

    plt.tight_layout()
    plt.show()


path = r'E:\second_year\会议\文章\数据\横切高程.txt'
num, space = 8, 500 / 1000  # 横切个数；与点之间的距离，单位为km
data = []
with open(path, 'r') as f:
    line = f.readline()
    line = f.readline()
    while line:
        tmp = line.split(',')
        data.append([int(tmp[2]), float(tmp[-2]), float(tmp[-1])])
        line = f.readline()

# 得到皱脊的字典
wrs_tmp, wrs = defaultdict(list), defaultdict(list)  # 字典中每个关键字是一条皱脊
# 取余num相同的是同一条皱脊
for x, y, z in data:
    key = int(x % num) if x % num else num
    wrs_tmp[key].append([x, y, z])

# 最终的皱脊结果
for key, values in wrs_tmp.items():
    mid = len(values) // 2
    wr = values[:mid][::-1] + values[mid + 1:]
    wrs[key] = wr

# 画图
# draw(wrs, path, space)

draw_selected(wrs, path, space)

# 分析
# 每个皱脊的高度
wr_mola, wr_sub = [], []
for i in wrs:
    value = wrs[i]
    wr_mola.append(np.array([row[1] for row in value]))
    wr_sub.append(np.array([row[2] for row in value]))

# for i in range(len(wr_mola)):
#     simpleplt([wr_mola[i]], title=i + 1)
#     simpleplt([wr_sub[i]], title=i + 1)

toe_sur_west = [8, 0, 0, 0, 0, 18, 4, 0]
toe_sur_east = [66, 76, 70, 74, 74, 70, 90, 81]
toe_sub_west = [5, 0, 11, 18, 11, 21, 27, 14]
toe_sub_east = [63, 86, 79, 80, 80, 79, 86, 71]
height_sur_west, height_sub_west, height_sur_east, height_sub_east, width_sur, width_sub, ele_offset = [], [], [], [], [], [], []
for i in range(len(wr_mola)):
    height_sur_west.append(np.max(wr_mola[i]) - wr_mola[i][toe_sur_west[i]])
    height_sur_east.append(np.max(wr_mola[i]) - wr_mola[i][toe_sur_east[i]])
    height_sub_west.append(np.max(wr_sub[i]) - wr_sub[i][toe_sub_west[i]])
    height_sub_east.append(np.max(wr_sub[i]) - wr_sub[i][toe_sub_east[i]])
    width_sur.append((toe_sur_east[i] - toe_sur_west[i]) * space)
    width_sub.append((toe_sub_east[i] - toe_sub_west[i]) * space)
    ele_offset.append(abs(wr_mola[i][toe_sur_west[i]] - wr_mola[i][toe_sur_east[i]]))

height_sur_avg = (np.array(height_sur_west) + np.array(height_sur_east)) / 2
height_sub_avg = (np.array(height_sub_west) + np.array(height_sub_east)) / 2

# 输出
for i in range(len(width_sub)):
    print(f"{ele_offset[i]:.1f}\t{width_sur[i]:.1f}\t{height_sur_west[i]:.1f}\t{height_sur_east[i]:.1f}\t{height_sur_avg[i]:.1f}\t{width_sur[i] / height_sur_avg[i] * 1000:.0f}")
    print(f"{ele_offset[i]:.1f}\t{width_sub[i]:.1f}\t{height_sub_west[i]:.1f}\t{height_sub_east[i]:.1f}\t{height_sub_avg[i]:.1f}\t{width_sub[i] / height_sub_avg[i] * 1000:.0f}")
