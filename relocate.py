from read_radargrams import *
from readroi import readroi
from findlatlon import lonlat
from re_dupli import redup


def findline(col, points_roi):
    # 目标：找到对应列的行值
    # 输入：列，所需层位的roi坐标信息
    # 输出：行
    cols = points_roi[:, 0]
    lines = points_roi[:, 1]
    index = np.where(cols == col)[0]
    if len(index) == 1:
        return lines[index[0]]
    else:
        if not len(index):
            # print(f'数组中没有{col}列')
            return 9999
        else:
            raise ValueError('Travel time repeat!')


def findmax(rgram, line, col, window):
    tmprg = rgram[line - window:line + window + 1, col]
    tmpmax = np.max(tmprg)
    indexmax = np.where(tmprg == tmpmax)[0]
    if len(indexmax) > 1:
        differences = np.abs(indexmax - line)
        closest_index = np.argmin(differences)
        num_max = indexmax[closest_index]
    else:
        num_max = indexmax[0]
    return line - window + num_max + 1


def relocate(path_radar, path_roi, filename, window=[10, 5, 5]):
    # 目标：根据能量对轨道的数据重定位
    # 输入：雷达数据的路径，roi.txt的路径，轨道编号（8位），重定位的范围（从地表向下的三元素列表，分别对应重定位窗口的大小）
    # 输出：roi数量，重定位之前的行列点（n维列表，每个元素为m*2的数组），重定位之后的行列点，经纬度，（走时，高程）
    # read data
    rgram = read_radar(path_radar, filename)

    # read roi points
    roi_num, roipoints = readroi(path_roi, filename)

    # Remove duplicate column; Take the average of lines
    roipoints_unique = redup(roipoints)

    # decide the layer order of ROIs
    minimum = np.zeros(roi_num)
    for i in range(roi_num):
        minimum[i] = min(roipoints_unique[i][:, 1])  # 1为行，找到roi的行最小值

    layerorder = np.argsort(minimum)

    # relocate each roi
    roi_relocated = []
    for i in range(roi_num):
        lindex = layerorder[i]
        lines = roipoints_unique[lindex][:, 1]
        cols = roipoints_unique[lindex][:, 0]
        lines_new = []
        for n in range(len(cols)):
            tmplines = findmax(rgram, lines[n] - 1, cols[n] - 1, window[i])  # 重定位：找到能量最大值的行
            if i == 0:  # surface
                lines_new.append(tmplines)
            else:  # subsurface
                try:
                    # if tmplines - findline(cols[n], roipoints_unique[layerorder[0]]) > 11:  # 去掉旁瓣影响
                    if tmplines - findline(cols[n], roi_relocated[0]) > 11:  # 去掉旁瓣影响
                        lines_new.append(tmplines)
                    else:
                        # print(f'列{cols[n]}地下走时为{tmplines}，地表走时为{findline(cols[n], roipoints_unique[layerorder[0]])}')
                        lines_new.append(9999)
                except:
                    lines_new.append(9999)  # 加上保持数量一致
                    print(filename, '地下', cols[n], '列不在地表中')

        roimatrix = np.array([cols, lines_new]).transpose()
        roi_relocated.append(roimatrix)

    # read geometry file
    coordinates_all = read_geom(path_radar, filename)
    # Find longitude and latitude
    roi_coordinates = lonlat(roi_num, roi_relocated, coordinates_all)

    # output
    return roi_num, roipoints, roi_relocated, roi_coordinates


if __name__ == "__main__":
    # 目标：计算path_roi文件夹内所有轨道的roi数，行列式值，经纬度
    path_radar = r'Z:\mars\sharad'
    path_roi = r'F:\ROI'
    roi_num_allfile, roi_relocate_before_allfile, roi_relocated_allfile, roi_coordinates_allfile = [], [], [], []
    window = [10, 7, 7]
    filelist = [i for i in os.listdir(path_roi) if i[-3:] == 'txt']  # 所有文件名
    for i in os.listdir(path_roi):
        if i[-3:] == 'txt':
            filename = i[:-4]
            data = relocate(path_radar, path_roi, filename, window)
            roi_num_allfile.append(data[0])
            roi_relocate_before_allfile.append(data[1])
            roi_relocated_allfile.append(data[2])
            roi_coordinates_allfile.append(data[3])

    np.savez(os.path.join(path_roi, '../data' + str(window[1]) + '.npz'), roi_num=roi_num_allfile, roi_relocate_before=roi_relocate_before_allfile, roi_relocated=roi_relocated_allfile, roi_coordinates=roi_coordinates_allfile)
