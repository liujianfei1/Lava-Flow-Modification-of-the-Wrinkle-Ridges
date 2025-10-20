from read_radargrams import *


def lonlat(roi_num, roi_relocated, coordinates_all):
    # 目标：计算界面点的经纬度
    # 输入：roi个数，重定位后的所有行列值，轨道的所有经纬度
    # 输出：列对应的经纬度（长度为roi_num的列表，每个元素为单层经纬度的数组，大小为m*2，m是每层的列数）
    roi_coord = []
    for n in range(roi_num):
        tmpcoord = []
        for col in range(len(roi_relocated[n])):
            colcoord = roi_relocated[n][col, 0]
            tmpcoord.append([coordinates_all[0][colcoord - 1], coordinates_all[1][colcoord - 1]])  # 经度和纬度 -180 ~ 180
        roi_coord.append(np.array(tmpcoord))

    return roi_coord
