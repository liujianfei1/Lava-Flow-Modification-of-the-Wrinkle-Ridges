import matplotlib
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")  # 关闭警告
matplotlib.use('TkAgg')
# 调整字号
plt.rcParams.update({
    'font.size': 16,  # 全局字体大小（轴标签、刻度、图例等）
    'axes.titlesize': 18,  # 坐标轴标题
    # 'axes.labelsize': 16,  # x/y轴标签
    'xtick.labelsize': 15,  # x轴刻度标签
    'ytick.labelsize': 15,  # y轴刻度标签
    'legend.fontsize': 15,  # 图例文字大小
})


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


def draw_all(productID_arr, loss_x_arr, loss_y_arr, slope_intercept_arr, loss_arr, r2_arr, pathsave):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()

    for i in range(4):
        productID = productID_arr[i]
        loss_x = loss_x_arr[i]
        loss_y = loss_y_arr[i]
        slope = slope_intercept_arr[i]
        loss = loss_arr[i]
        r2 = r2_arr[i]

        ax = axs[i]
        draw(productID, loss_x, loss_y, slope, r2=r2, loss=loss, ax=ax)

        # 设置坐标轴标签显示条件
        if i in [0, 2]:  # 第1列显示y轴
            ax.set_ylabel('Normalized power (dB)')
        else:
            ax.set_ylabel('')

        if i in [2, 3]:  # 第2行显示x轴
            ax.set_xlabel('Time delay ($\mu$s)')
        else:
            ax.set_xlabel('')

    # 去除多余子图（如不足4张）
    for j in range(len(productID_arr), 4):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(os.path.join(pathsave, 'v4.png'), dpi=300)
    plt.close()


def draw(productID, loss_x, loss_y, slope_intercept, r2=None, loss=None, ax=None):
    x_new = np.linspace(min(loss_x) - 0.1, max(loss_x) + 0.1, 1000)
    polynomial = np.poly1d(slope_intercept)
    y_new = polynomial(x_new)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x_new, y_new, '-', linewidth=2, c='k', label='fit')
    # ax.scatter(loss_x, loss_y, marker='x', color='k', label='loss')
    ax.scatter(loss_x, loss_y, marker='o', color='skyblue', label='loss', alpha=0.9)

    # ax.set_xlabel('Time delay ($\mu$s)')
    # ax.set_ylabel('Normalized power (dB)')
    # 调整 x 轴范围
    ax.set_xlim(min(loss_x) - 0.3, max(loss_x) + 0.3)
    ax.set_title(productID)

    # 注释 R² 和 loss tangent
    if r2 is not None and loss is not None:
        textstr = f'$R^2$ = {r2:.2f}\nLoss tangent = {loss:.3f}'
        ax.text(0.6, 0.95, textstr, transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

    return ax


def groupdata(data):
    data_sorted = data[np.argsort(data[:, 0])]  # 按第一列排序

    groups = []
    current_group = [data_sorted[0]]  # 初始化第一组

    for i in range(1, data.shape[0]):
        # 检查当前行与前一行是否连续
        if data_sorted[i][0] - data_sorted[i - 1][0] == 1:
            current_group.append(data_sorted[i])  # 添加到当前组
        else:
            groups.append(np.array(current_group))  # 完成一组
            current_group = [data_sorted[i]]  # 新的一组

    groups.append(np.array(current_group))

    return groups


if __name__ == "__main__":
    pathsave_good = r'E:\second_year\会议\文章\图片\final\7损失正切'
    if not os.path.exists(pathsave_good):
        os.makedirs(pathsave_good)
    t, c = 0.0375, 299.792458
    path_roi = r'F:\亚马逊平原皱脊\数据验证\ROI'
    path_data = r'E:\second_year\会议\文章\数据'
    filelist_all = [i for i in os.listdir(path_roi) if i[-3:] == 'txt']  # 所有文件名
    filelist = ["00484201", "00668801", "03735802", "06229401"]  # 挑选的四条
    data = np.load(os.path.join(path_data, 'data7.npz'), allow_pickle=True)
    roi_num_allfile, roi_relocated_allfile = data['roi_num'], data['roi_relocated']
    loss_arr, std_arr, productID_arr, r2_arr, slope_arr, slope_std_arr = [], [], [], [], [], []
    loss_x_arr, loss_y_arr, slope_intercept_arr = [], [], []
    for productID in filelist:
        k = filelist_all.index(productID + '.txt')
        # 打开img
        file = 's_' + productID + '_rgram.lbl'
        path_rgram = os.path.join(r'Z:\mars\sharad\rgram', file)
        data_rgram = rasterio.open(path_rgram)
        img_rgram = data_rgram.read()[0]  #(1,3600,m) .transpose((1, 2, 0))

        roi_num, roi_relocated = roi_num_allfile[k], roi_relocated_allfile[k]
        layernum = len(roi_relocated)
        if layernum > 1:  # 有地下界面
            suf = roi_relocated[0]
            sub_all = roi_relocated[1]
            sub_group = groupdata(sub_all)
            for sub in [sub_all]:  # sub_group [sub_all]
                loss_x, loss_y = [], []
                cols = sub[:, 0]
                for i in range(len(cols)):
                    subtime = sub[:, 1][i]
                    if subtime != 9999:
                        surtime = findline(cols[i], suf)
                        P0 = img_rgram[:, cols[i] - 1][surtime - 1]  # 因为python的矩阵从零开始，geo是从一开始，所以要-1
                        P = img_rgram[:, cols[i] - 1][subtime - 1]
                        loss_x.append((subtime - surtime) * t)
                        loss_y.append(math.log(P / P0, 10) * 10)  # 乘10是为了图像斜率更清楚
                if len(loss_y) < 3:
                    continue
                slope_intercept, cov = np.polyfit(loss_x, loss_y, 1, cov=True)
                y_pred = np.poly1d(slope_intercept)(loss_x)
                X = sm.add_constant(loss_x)
                model = sm.OLS(loss_y, X)
                results = model.fit()
                # print(results.summary())
                wald_test = results.wald_test('x1 = 0')
                p_value = wald_test.pvalue
                r2 = r2_score(loss_y, y_pred)
                std_devs = np.sqrt(np.diag(cov)[0])  # 斜率的标准差，(小于斜率10%的比较可信，抛弃，改用决定系数)
                if (slope_intercept[0] > 0 or r2 < 0.3):  # r2 < 0.3 p_value > 0.05
                    loss_cal = math.sqrt((2 * (15 * math.log(10**(slope_intercept[0] / 10)) / (4 * math.pi * c))**2 + 1)**2 - 1)
                    print('不好:', productID + ': 点个数{:d} 斜率{:.2f}±{:.2f} 损失正切{:.3f} p值{:.2f} 决定系数{:.2f}'.format(len(loss_y), slope_intercept[0], std_devs, loss_cal, p_value, r2))
                    # draw(productID, loss_x, loss_y, slope_intercept, pathsave_bad)
                    continue
                loss_cal = math.sqrt((2 * (15 * math.log(10**(slope_intercept[0] / 10)) / (4 * math.pi * c))**2 + 1)**2 - 1)
                k = 15 * math.log(10) / (4 * math.pi * c * 10)
                std_cal = k * math.sqrt(k) * abs(4 * k**2 * slope_intercept[0]**3 + 2 * slope_intercept[0]) / math.sqrt(k**2 * slope_intercept[0]**4 + slope_intercept[0]**2) * std_devs
                slope_arr.append(slope_intercept[0])
                slope_std_arr.append(std_devs)
                loss_arr.append(loss_cal)
                std_arr.append(std_cal)
                productID_arr.append(productID)
                r2_arr.append(r2)
                print('好:', productID + ': 点个数{:d} 斜率{:.2f}±{:.2f} 损失正切{:.3f}±{:.3f} p值{:.2f} 决定系数{:.2f}'.format(len(loss_y), slope_intercept[0], std_devs, loss_cal, std_cal, p_value, r2))
                loss_x_arr.append(loss_x)
                loss_y_arr.append(loss_y)
                slope_intercept_arr.append(slope_intercept)
    draw_all(productID_arr, loss_x_arr, loss_y_arr, slope_intercept_arr, loss_arr, r2_arr, pathsave_good)
    print(loss_arr, min(loss_arr), max(loss_arr))
    print(std_arr)
