import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter
import matplotlib
from read_radargrams import read_radar, read_geom
from readroi import readroi
from caltime import layer_diff, findline

matplotlib.use('TkAgg')
# 调整字号
plt.rcParams.update({
    'font.size': 13,  # 全局字体大小（轴标签、刻度、图例等）
    'axes.titlesize': 15,  # 坐标轴标题
    'xtick.labelsize': 12,  # x轴刻度标签
    'ytick.labelsize': 12,  # y轴刻度标签
    'legend.fontsize': 12,  # 图例文字大小
})

Y_OFFSET = 0.1


# ======================== 工具函数 ========================
def scale(data):
    img_scale = np.log10(data + 1e-30)
    img_valid = img_scale[data != 0]
    p10 = np.percentile(img_valid, 10)
    m = 255 / (img_valid.max() - p10)
    b = -p10 * m
    img_map = (m * img_scale) + b
    img_map[img_map < 0] = 0
    img_uint = img_map.astype(np.uint8)
    return img_uint


def distance(lon1, lat1, lon2, lat2):
    return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)


def closest_index(lon1, lat1, lon_list, lat_list):
    return min(range(len(lon_list)), key=lambda i: distance(lon1, lat1, lon_list[i], lat_list[i]))


def plot_thick(ax, thick, sub_lines, startx, endx, start_index, end_index, offset):
    smoothed_thick = savgol_filter(thick, window_length=15, polyorder=2)
    ax.scatter(sub_lines * 0.46, thick, s=7, label='thickness')
    ax.plot(sub_lines * 0.46, smoothed_thick, color='pink', linewidth=3, alpha=0.6, label="thickness fitting")
    ax.set_xlim(startx * 0.46, endx * 0.46)
    lenyoffset = (max(thick[start_index:end_index + 1]) - min(thick[start_index:end_index + 1])) * Y_OFFSET
    ymin = min(thick[start_index:end_index + 1]) - lenyoffset
    ymax = max(thick[start_index:end_index + 1]) + lenyoffset
    ymin_round = math.ceil(ymin / 10) * 10
    ymax_round = math.floor(ymax / 10) * 10
    ax.set_ylim(ymin, ymax)

    # 只保留 y 轴的最小值和最大值
    ax.set_yticks([ymin_round, ymax_round])
    ax.set_yticklabels([str(ymin_round), str(ymax_round)])
    ax.set_ylabel('T (m)', rotation=0, labelpad=25, va='center')


def plot_elevation(ii, ax, roipoints, cols_plt, start_index, end_index, startx, endx, offset):
    ax.axis('off')  # 隐藏父图坐标轴
    ep, c, t = 7.5, 300, 0.0375
    suf_lines = roipoints[0][:, 0]
    suf_cols = roipoints[0][:, 1]
    sub_lines = roipoints[1][:, 0]
    suf_ele = -(suf_cols - 1800) * t * c / 2
    cols = roipoints[1][:, 0]
    layerup = np.array([roipoints[0][:, 0], suf_ele]).transpose()
    thick = layer_diff(roipoints[1], roipoints[0]) * t * c / 2 / math.sqrt(ep)

    elva_sub = []
    for n in range(len(cols)):
        col = cols[n]
        elevup = findline(col, layerup)
        elva_sub.append(elevup - thick[n])

    # 创建子图内的两个子图
    ax21 = ax.inset_axes([0, 0.75, 1, 0.25])
    ax22 = ax.inset_axes([0, 0, 1, 0.75])

    # 厚度子图
    plot_thick(ax21, thick, sub_lines, startx, endx, start_index[1], end_index[1], offset)

    # 表面与地下高程子图
    ax22.scatter(suf_lines * 0.46, suf_ele, s=7, label='surface elevation')
    ax22.scatter(sub_lines * 0.46, elva_sub, s=7, label='subsurface elevation')
    smoothed_sur = savgol_filter(suf_ele, window_length=15, polyorder=2)
    ax22.plot(suf_lines * 0.46, smoothed_sur, color='g', linewidth=3, alpha=0.5, label="surface fitting")
    smoothed_sub = savgol_filter(elva_sub, window_length=15, polyorder=2)
    ax22.plot(sub_lines * 0.46, smoothed_sub, color='y', linewidth=3, alpha=0.5, label="subsurface fitting")

    ax22.set_xlim(startx * 0.46, endx * 0.46)
    dx = 1 * 0.46
    xticks = np.arange(startx * 0.46, endx * 0.46, dx)
    # xtick_labels = [f"{abs(xtick - offset * 0.46 + 1):.0f}" for xtick in xticks]
    # ax22.set_xticks(xticks)
    # ax22.set_xticklabels(xtick_labels)
    # 只保留最左端和最右端
    xticks_new = [xticks[0], offset * 0.46, xticks[-1]]
    xtick_labels = [f"{int(np.floor(abs(tick - offset * 0.46)))}" for tick in xticks_new]

    ax22.set_xticks(xticks_new)
    ax22.set_xticklabels(xtick_labels)

    ax22.set_xlabel('Relative Distance from WR top (km)')
    ax22.set_ylabel('H (m)', labelpad=2)

    lenyoffset = (max(suf_ele[start_index[0]:end_index[0] + 1]) - min(elva_sub[start_index[1]:end_index[1] + 1])) * Y_OFFSET
    ax22.set_ylim(min(elva_sub[start_index[1]:end_index[1] + 1]) - lenyoffset, max(suf_ele[start_index[0]:end_index[0] + 1]) + lenyoffset)

    if ii == 3:
        ax22.set_ylim(min(elva_sub[start_index[1]:end_index[1] + 1]) - lenyoffset * 3,
                      max(suf_ele[start_index[0]:end_index[0] + 1]) + lenyoffset)
        handles_21, labels_21 = ax21.get_legend_handles_labels()
        handles_22, labels_22 = ax22.get_legend_handles_labels()
        # ax22.legend(handles_21 + handles_22, labels_21 + labels_22, ncol=2, loc='best', framealpha=0.8)
    elif ii == 2:
        ax22.invert_xaxis()
        ax21.invert_xaxis()

    for a in [ax21, ax22]:
        for i in cols_plt:
            a.axvline(i * 0.46, color='r', linestyle='--', alpha=0.7, label='WR top')


# ======================== 主程序 ========================
if __name__ == "__main__":
    path = r'Z:\mars\sharad'
    namelist = ["00685902", "00786101", "05671001", "00721501"]
    lonlist = [[-162.484051, -162.503328, -162.523308], [-162.590583], [-162.673672], [-162.666056, -162.754191, -162.778823]]
    latlist = [[34.331293, 34.197507, 34.058844], [33.998503], [33.978093], [33.987137, 33.372854, 33.200026]]
    path_roi = r"F:\亚马逊平原皱脊\总数据\校正ROI_v1"

    fig, axes = plt.subplots(len(namelist), 1, figsize=(5, 8.5))  # 单列
    axes = np.array(axes).flatten()
    labels = ['b1', 'b2', 'b3', 'b4']

    for ii in range(len(namelist)):
        print(f"Processing {namelist[ii]} ({ii + 1}/{len(namelist)})")
        name = namelist[ii]
        rgramMap = scale(read_radar(path, name))
        lon_list, lat_list = read_geom(path, name)
        lon, lat, close_col = lonlist[ii], latlist[ii], []
        for i in range(len(lon)):
            close_col.append(closest_index(lon[i], lat[i], lon_list, lat_list))
        offset = close_col[0]
        num1, num2 = 145, 144
        startx, endx = (offset - num1) * 0.46, (offset + num2) * 0.46
        startx = round(startx / 0.46)
        endx = round(endx / 0.46)
        filename = name + '_new'
        roi_num, roipoints = readroi(path_roi, filename)
        roipoints_minus1 = [arr - 1 for arr in roipoints]
        new_roipoints_minus1 = [np.array([x[:, 0], x[:, 1]]).transpose() for x in roipoints_minus1]
        start_index = [np.where(new_roipoints_minus1[0][:, 0] == startx)[0][0],
                       np.where(new_roipoints_minus1[1][:, 0] == startx)[0][0]]
        end_index = [np.where(new_roipoints_minus1[0][:, 0] == endx)[0][0],
                     np.where(new_roipoints_minus1[1][:, 0] == endx)[0][0]]

        ax = axes[ii]
        plot_elevation(ii, ax, new_roipoints_minus1, close_col, start_index, end_index, startx, endx, offset)
        ax.text(-0.15, 1.3, labels[ii], transform=ax.transAxes, fontweight='bold', color='black',
            va='top', ha='left')

        # 隐藏 x 轴名和刻度（除了最后一个子图）
        ax22_list = ax.get_children()
        for child in ax22_list:
            if isinstance(child, matplotlib.axes._axes.Axes):  # 处理 inset 子图
                if ii != len(namelist) - 1:
                    child.set_xticks([])
                    child.set_xticklabels([])
                    child.set_xlabel('')  # 隐藏轴名
                else:
                    # 保留最后一个子图的 x 轴，如果不需要也可以注释掉
                    child.set_xlabel('Relative Distance from WR top (km)')  # 最后一个子图显示轴名

    plt.tight_layout()
    plt.savefig(r"E:\second_year\会议\文章\图片\final\8皱脊地下高程\四个雷达高程v5.svg",
        format='svg', dpi=300, bbox_inches='tight')
