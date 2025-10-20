import os
import re
import numpy as np


# 目标：得到roi的行列值
# 输入：roi.txt的路径和文件基本名
# 输出：轨道层数roi_num，长度为roi_num的列表，每个元素为m*2的数组，第0维对应列值，第1维对应行值
def readroi(path_roi, filename):
    #find roi information： number of ROI; NUMBER OF POINTS IN EACH ROI
    path_ID = os.path.join(path_roi, filename + '.txt')
    points_num = []
    with open(path_ID, 'r') as f:  # 读取层位的行列数，注意记录与实际差1（python是从0开始的），所以这里有个减1
        lines = f.readlines()

    for n, line in zip(range(len(lines)), lines):
        if line[0] == ';':
            if line[:-2] == '; Number of ROIs: ':
                roi_num = int(line[-2])
            elif line[:10] == '; ROI npts':
                points_num.append(int(re.split(':', line)[-1]))
        else:
            print(f'{filename} found {roi_num} ROIs, with {points_num} points, respectively.')
            break
    roi_line_num = [n]
    for i in range(roi_num):
        new_line = roi_line_num[i] + points_num[i] + 1
        roi_line_num.append(new_line)
    points = []
    for i in range(roi_num):
        roi_tmp = []
        for line in lines[roi_line_num[i]:roi_line_num[i + 1] - 1]:
            tmp = list(filter(None, re.split(' ', line.strip())))
            roi_tmp.append([int(tmp[1]), int(tmp[2])])
        points.append(np.array(roi_tmp))
    # 输出的points是一个roi_num长度的列表，其中每一个元素是n*2的数组，其中每一行为对应roi的X和Y（列和行）
    return roi_num, points