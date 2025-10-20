import math

from relocate import *


def layer_diff(points, pointsup):
    # 目标：计算两层的差
    # 输入：下层，上层，类型为n*2的数组，列值和行值
    # 输出：下层对应的差值，类型为数组
    cols, lines, diff = points[:, 0], points[:, 1], []
    for n in range(len(cols)):
        col = cols[n]
        line = lines[n]
        lineup = findline(col, pointsup)
        if line == 9999:
            diff.append(9999)
        else:
            diff.append(line - lineup)
    return np.array(diff)


def calttime(roi_num, roi_relocated, t=0.0375):
    # 目标：计算所有层的走时差，只能计算两层以上的轨道
    # 输入：roi数，重定位后的所有行列值，每个像素点对应的走时
    # 输出：所有层对应的双程走时（长度为roi_num-1的列表，每个元素为下层与上层走时差的数组）
    travel_time = []
    for i in range(roi_num - 1):
        travel_time.append(layer_diff(roi_relocated[i + 1], roi_relocated[i]) * t)

    return travel_time


def calelev_sur(points_roi, t=0.0375, c=299.792458):
    # 计算地表高程
    cols = points_roi[:, 1]
    return -(cols - 1800) * t * c / 2


def calelev_ly(roi_num, roi_relocated, t=0.0375, c=299.792458, ep=[7.50, 11.14]):
    # 目标：计算各层的高程，包括地表
    # 输入：roi数，重定位后的所有行列值，各层介电常数
    # 输出：所有点对应的高程（长度为roi_num的列表，每个元素为单层经纬度的数组，大小为m*2，m是每层的列数）
    roi_elev = []
    for i in range(roi_num):
        if i == 0:
            roi_elev.append(calelev_sur(roi_relocated[i]))
        else:
            elva_sub = []
            cols = roi_relocated[i][:, 0]
            layerup = np.array([roi_relocated[i - 1][:, 0], roi_elev[-1]]).transpose()  # 上一层的列和高程
            thick = layer_diff(roi_relocated[i], roi_relocated[i - 1]) * t * c / 2 / math.sqrt(ep[i - 1])  # 本层厚度
            for n in range(len(cols)):
                col = cols[n]
                elevup = findline(col, layerup)
                elva_sub.append(elevup - thick[n])
            roi_elev.append(np.array(elva_sub))
    return roi_elev


def writefile(path, name, n):
    with open(os.path.join(path, name), 'w') as f:
        # f.write("longitude latitude " + typestr + "\n")
        if n:
            f.write("orbit longitude latitude elevation traveltime\n")
        else:
            f.write("orbit longitude latitude elevation\n")


def writefilea(path, name, name_orbit, lines, lons, lats, elevs, ttimes=[], typestr=''):
    with open(os.path.join(path, name), 'a') as f:
        # f.write("longitude latitude " + typestr + "\n")
        if len(ttimes):
            # f.write("longitude latitude elevation traveltime\n")
            for i in range(len(lines)):
                if lines[i] <= 3600:
                    f.write(f"{name_orbit} {lons[i]} {lats[i]} {elevs[i]} {ttimes[i]}\n")
        else:
            # f.write("longitude latitude elevation\n")
            for i in range(len(lines)):
                f.write(f"{name_orbit} {lons[i]} {lats[i]} {elevs[i]}\n")


if __name__ == "__main__":
    # 输出为层位的经纬度，分为三层：layer_0是地表，layer_1是次表层，layer_2是地下第二层
    # layer_1文件中有五个变量，分别为轨道数、经度、纬度、高程、层位单程走时
    # t, c, ep = 0.0375, 299.792458, [4.66, 11.14]
    path_w = r'E:\second_year\会议\文章\数据'
    filelist = [i for i in os.listdir(r'F:\亚马逊平原皱脊\数据验证\ROI') if i[-3:] == 'txt']  # 所有文件名
    data = np.load(os.path.join(path_w, 'data7.npz'), allow_pickle=True)
    roi_num_allfile, roi_relocated_allfile, roi_coordinates_allfile, roi_ttime_allfile, roi_elev_allfile = data['roi_num'], data['roi_relocated'], data['roi_coordinates'], [], []
    for k in range(len(roi_num_allfile)):
        roi_num, roi_relocated, roi_coordinates = roi_num_allfile[k], roi_relocated_allfile[k], roi_coordinates_allfile[k]
        layernum = len(roi_relocated)
        if layernum == 1:  # 只有地表
            roi_elev_allfile.append([calelev_sur(roi_relocated[0])])
        elif layernum > 1:  # 有地表和地下层位
            roi_ttime_allfile.append(calttime(roi_num, roi_relocated))
            roi_elev_allfile.append(calelev_ly(roi_num, roi_relocated))

    # 写文件
    name_file = 'layer_'
    # 写文件头
    for j in range(max(roi_num_allfile)):
        name_w = name_file + str(j) + '.txt'
        writefile(path_w, name_w, j)
    subcount = 0  # 计地下层位的数
    for k in range(len(roi_num_allfile)):
        roi_num, roi_relocated, roi_coordinates, roi_elev, name_orbit = roi_num_allfile[k], roi_relocated_allfile[k], roi_coordinates_allfile[k], roi_elev_allfile[k], filelist[k][:-4]
        if roi_num > 1:
            roi_ttime = roi_ttime_allfile[subcount]
            subcount += 1
        for i in range(roi_num):
            name_w = name_file + str(i) + '.txt'
            # 地表列，行，经度，纬度，高程
            cols = roi_relocated[i][:, 0]
            lines = roi_relocated[i][:, 1]
            lons = roi_coordinates[i][:, 0]
            lats = roi_coordinates[i][:, 1]
            elevs = roi_elev[i]
            if i != 0:
                ttimes = roi_ttime[i - 1]
                writefilea(path_w, name_w, name_orbit, lines, lons, lats, elevs, ttimes)
            else:
                writefilea(path_w, name_w, name_orbit, lines, lons, lats, elevs)
