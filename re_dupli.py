import numpy as np


def redup(roipoints):
    # 目标：判断行列值中的列是否有重复值，有就去重，取该列所有行的平均值
    # 输入：轨道所有roi的行列值
    # 输出：去重后的行列值，长度为roi_num的列表，每个元素为m*2的数组，第0维对应列值，第1维对应行值
    redup_points = []
    for i in range(len(roipoints)):
        points = roipoints[i]
        lines = points[:, 1]
        cols = points[:, 0]
        col_len_unique = np.unique(cols).size
        if cols.size > col_len_unique:
            tmpunique, tmpsums, tmpcounts = np.empty([col_len_unique, 2], dtype=int), {}, {}

            for line, col in zip(lines, cols):
                if col not in tmpsums:
                    tmpsums[col] = 0
                    tmpcounts[col] = 0
                tmpsums[col] += line
                tmpcounts[col] += 1
            for k, key in enumerate(tmpsums):
                tmpunique[k, 0] = int(key)
                tmpunique[k, 1] = round(tmpsums[key] / tmpcounts[key])
            redup_points.append(np.array(tmpunique))
        else:
            redup_points.append(points)

    return redup_points
