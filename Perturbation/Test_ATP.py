import numpy as np
import pickle
import random
import pandas as pd
from collections import defaultdict
from scipy.stats import kendalltau

# 2713  1167   1081   4000
# Path = 'E:\Code\Trace\dataset\CHI\ATP_CHI\\'
Path = 'E:\Code\Trace\dataset\CLE\ATP_CLE\\'
# Path = 'E:\\Code\\Trace\\dataset\\NYC\\ATP_NYC\\'
# Path = 'E:\\Code\\Trace\\dataset\\CPS\\ATP_CPS\\'

with open(Path + 'traj_list.pkl', 'rb') as traj_list_read:
    traj_list = pickle.load(traj_list_read)
# print(len(traj_list))

with open(Path + 'longitude_dict_file.pkl', 'rb') as longitude_dict_read:
    longitude_dict = pickle.load(longitude_dict_read)

with open(Path + 'latitude_dict_file.pkl', 'rb') as latitude_dict_read:
    latitude_dict = pickle.load(latitude_dict_read)

with open(Path + 'str2index_dict_file.pickle', 'rb') as str2index_dict_read:
    str2index_dict = pickle.load(str2index_dict_read)

with open(Path + 'index2str_dict_file.pickle', 'rb') as index2str_dict_read:
    index2str_dict = pickle.load(index2str_dict_read)

with open(Path + 'np_poi_lat_lon_distance_matrix_file.pickle', 'rb') as poi_distance_matrix_read:
    poi_distance_matrix = pickle.load(poi_distance_matrix_read)

with open(Path + 'poi_angel_num_matrix_6_file.pickle', 'rb') as poi_angel_num_matrix_6_read:
    poi_angel_num_matrix_6 = pickle.load(poi_angel_num_matrix_6_read)

with open(Path + 'poi_angel_num_matrix_4_file.pickle', 'rb') as poi_angel_num_matrix_4_read:
    poi_angel_num_matrix_4 = pickle.load(poi_angel_num_matrix_4_read)

with open(Path + 'poi_angel_num_matrix_2_file.pickle', 'rb') as poi_angel_num_matrix_2_read:
    poi_angel_num_matrix_2 = pickle.load(poi_angel_num_matrix_2_read)

with open(Path + 'poi_angel_num_matrix_12_file.pickle', 'rb') as poi_angel_num_matrix_12_read:
    poi_angel_num_matrix_12 = pickle.load(poi_angel_num_matrix_12_read)

with open(Path + 'traj_center_list_file.pkl', 'rb') as traj_center_list_read:
    traj_center_list = pickle.load(traj_center_list_read)

delta_poi_distance_matrix = poi_distance_matrix.max()

# 对轨迹点方向划分的粒度
def get_reversed_num_direc_of_traj(perturbed_direc_of_traj, direc_num=4):
    assert direc_num in [2, 4, 6, 12]
    reversed_perturbed_num_direc_of_traj = np.array(perturbed_direc_of_traj)
    if direc_num == 4:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 2) % 4
    elif direc_num == 2:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 1) % 2
    elif direc_num == 6:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 3) % 6
    elif direc_num == 12:
        reversed_perturbed_num_direc_of_traj = (reversed_perturbed_num_direc_of_traj + 6) % 12
    else:
        print('get_reversed_num_direc_of_traj wrong!')
    reversed_perturbed_num_direc_of_traj = reversed_perturbed_num_direc_of_traj.tolist()

    return reversed_perturbed_num_direc_of_traj


def kRR(value, vals, epsilon):
    """
    the k-random response
    :param value: current value
    :param values: the possible value
    :param epsilon: privacy budget
    :return:
    """
    values = vals.copy()
    p = np.e ** epsilon / (np.e ** epsilon + len(values) - 1)
    if np.random.random() < p:
        return value
    values.remove(value)
    return values[np.random.randint(low=0, high=len(values))]


def square_wave_mechanism(poi_idx, R, epsilon):
    delta = max(poi_distance_matrix[poi_idx])
    t = R / delta
    b = (epsilon * (np.e ** (epsilon)) - np.e ** epsilon + 1) / (
            2 * np.e ** (epsilon) * (np.e ** epsilon - 1 - epsilon))
    x = random.uniform(0, 1)
    if x < (2 * b * np.e ** epsilon) / (2 * b * np.e ** epsilon + 1):
        perturbed_t = random.uniform(- b + t, b + t)
    else:
        x_1 = random.uniform(0, 1)
        if x_1 <= t:
            perturbed_t = random.uniform(- b, - b + t)
        else:
            perturbed_t = random.uniform(b + t, b + 1)
    perturbed_t = (perturbed_t + b) / (2 * b + 1)
    perturbed_t = perturbed_t * delta

    return perturbed_t

# 根据划分方向 找到轨迹点的扰动值域
def find_this_direc_support_of_this_poi(this_poi, this_poi_direc, poi_angel_num_matrix_adap):
    this_poi_direc_num_line = poi_angel_num_matrix_adap[this_poi]
    this_direc_support_array = np.argwhere(this_poi_direc_num_line == this_poi_direc).reshape(-1)

    return this_direc_support_array

# 找到所有满足条件的轨迹点
def find_all_support_of_this_poi(this_poi, this_poi_direc, poi_angel_num_matrix_adap):
    this_poi_direc_num_line = poi_angel_num_matrix_adap[this_poi]
    this_direc_support_array = np.argwhere(this_poi_direc_num_line >= 0).reshape(-1)

    return this_direc_support_array

# 对于半径R的后置处理
def post_process_for_R(traj_center_nearest_poi, perturbed_R, epsilon_for_d1):
    delta = max(poi_distance_matrix[traj_center_nearest_poi])

    b = (epsilon_for_d1 * (np.e ** (epsilon_for_d1)) - np.e ** epsilon_for_d1 + 1) / (
            2 * np.e ** (epsilon_for_d1) * (np.e ** epsilon_for_d1 - 1 - epsilon_for_d1))
    t_star = (perturbed_R / delta) * (2 * b + 1) - b
    t_list = np.arange(0, 1.1, step=0.1)

    res_1_list = []
    res_2_list = []
    t_idx_list = []

    for t_idx, t in enumerate(t_list):
        d = (2 * b * (np.e ** (epsilon_for_d1) - 1) * t) / (2 * b * np.e ** epsilon_for_d1 + 1) + (1 + 2 * b) / (
                2 * (2 * b * np.e ** (epsilon_for_d1) + 1)) - t
        v = b ** 2 / 3 + ((2 * b + 1) * (b + 1 - 3 * t ** 2)) / (
                3 * (2 * b * np.e ** epsilon_for_d1 + 1)) - d ** 2 - 2 * d * t

        l = t - b
        r = t + b

        if t_star <= r and t_star >= l:
            t_idx_list.append(t_idx)

    t_cali_list = []
    for t_idx in t_idx_list:
        t_cali_list.append(t_list[t_idx])

    t_l = min(t_cali_list)
    t_r = max(t_cali_list)
    t_l = (t_l + b) * delta / (2 * b + 1)
    t_r = (t_r + b) * delta / (2 * b + 1)

    for poi_idx, poi_distance in enumerate(poi_distance_matrix[traj_center_nearest_poi]):
        if t_l < 0:
            t_l = 0
        if poi_distance <= t_r and poi_distance >= t_l:
            res_1_list.append(poi_distance * np.e ** epsilon_for_d1 / (2 * b * np.e ** epsilon_for_d1 + 1))
        else:
            res_2_list.append(poi_distance * 1 / (2 * b * np.e ** epsilon_for_d1 + 1))

    if len(res_1_list) > 0:
        t_star_cali = (sum(res_1_list) + sum(res_2_list)) \
                      / (len(res_1_list) * np.e ** epsilon_for_d1 / (2 * b * np.e ** epsilon_for_d1 + 1) + len(
            res_2_list) * 1 / (2 * b * np.e ** epsilon_for_d1 + 1))
    else:
        t_star_cali = sum(poi_distance_matrix[traj_center_nearest_poi]) / (poi_distance_matrix.shape[0] - 1)

    return t_star_cali


def sigmoid(x):
    s = 1 / (1 + np.exp(- x))
    return s

#
def ATP_mechanism(traj, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list, epsilon_direct_rate=0.75,
                  kRR_exp_kRR=False, traj_idx=None):
    # 轨迹点集合  隐私预算  与轨迹点方向相关的数据结构 隐私预算分配比例

    epsilon_traj_center_rate = 0.25
    assert traj_idx != None
    # 方向扰动
    epsilon_direction = epsilon * epsilon_direct_rate * 0.75 / (len(traj) - 1)
    # 锚点扰动
    epsilon_traj_center = epsilon * epsilon_traj_center_rate * 0.25
    epsilon_for_d1 = epsilon * epsilon_traj_center_rate * 0.75
    # 对每个轨迹点扰动
    epsilon_per_poi = (epsilon - epsilon * 0.75 * epsilon_direct_rate - epsilon * epsilon_traj_center_rate) / (
        len(traj))

    traj_array = []
    for this_poi_str in traj:
        traj_array.append(str2index_dict[this_poi_str])

    # 锚点扰动
    traj_center_nearest_poi = str2index_dict \
        [Exp_mechanism([traj_center_list[traj_idx]], epsilon_traj_center)[0]]

    R_2 = max(poi_distance_matrix[traj_center_nearest_poi][traj_array])

    # 扰动锚点的半径R
    R = square_wave_mechanism(traj_center_nearest_poi, R_2, epsilon_for_d1)

    assert R >= 0

    traj_center_dist_matrix = poi_distance_matrix[traj_center_nearest_poi]
    avg_dist = post_process_for_R(traj_center_nearest_poi, R, epsilon_for_d1)
    # 调整半径R
    if R < avg_dist:
        R = R + (avg_dist - R) * np.e ** (- epsilon_for_d1) * sigmoid((avg_dist - R) / (2 * avg_dist))
    else:
        R = R - (R - avg_dist) * np.e ** (- epsilon_for_d1) * sigmoid(
            (R - avg_dist) / (2 * (max(poi_distance_matrix[traj_center_nearest_poi]) - avg_dist)))

    mask = np.argwhere(traj_center_dist_matrix <= R)

    poi_list_for_dist = []
    poi_list_for_direc_dist = []
    perturbed_traj = []
    # 轨迹分为两个副本
    # poi_list_for_dist存放枢轴点   poi_list_for_direc_dist 存放非枢轴点
    if kRR_exp_kRR == False:
        for i, this_poi in enumerate(traj_array):
            if i % 2 == 0:
                poi_list_for_dist.append(this_poi)
            else:
                poi_list_for_direc_dist.append(this_poi)
    else:
        for i, this_poi in enumerate(traj_array):
            if i % 2 == 1:
                poi_list_for_dist.append(this_poi)
            else:
                poi_list_for_direc_dist.append(this_poi)
    perturbed_poi_list_for_dist = []
    for i, this_poi in enumerate(poi_list_for_dist):
        x = find_all_support_of_this_poi(this_poi, 1, poi_angel_num_matrix_adap)
        y = np.intersect1d(mask, x).reshape(-1)
        # 利用指数机制扰动枢轴点
        delta_of_this_poi_direc = delta_poi_distance_matrix
        over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y] / (2 * delta_of_this_poi_direc)))

        total = over.sum()

        pdf = over / total
        cdf = pdf.copy()
        for j in range(1, pdf.shape[0]):
            cdf[j] += cdf[j - 1]
        rand_uni_num = np.random.random()

        perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
        perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
        perturbed_poi_index = int(perturbed_poi_index)

        perturbed_poi_list_for_dist.append(perturbed_poi_index)
    # 扰动非枢轴点
    # print('traj_array:',traj_array)
    # print('perturbed_poi_list_for_dist:',perturbed_poi_list_for_dist)
    # print('poi_list_for_direc_dist:',poi_list_for_direc_dist)
    perturbed_poi_list_for_direc_dist = []
    if len(traj_array) % 2 == 1:
        for i, this_poi in enumerate(poi_list_for_direc_dist):
            # print('this_poi:',i,this_poi)
            if kRR_exp_kRR == False:
                perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                              poi_angel_num_adap_list, epsilon_direction)
                perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i + 1]],
                                               poi_angel_num_adap_list, epsilon_direction)
                # print(perturbed_poi_pre_direc,perturbed_poi_post_direc)

                reversed_perturbed_poi_post_direc = \
                    get_reversed_num_direc_of_traj([perturbed_poi_post_direc], direc_num=len(poi_angel_num_adap_list))[
                        0]

                this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       perturbed_poi_pre_direc,
                                                                                       poi_angel_num_matrix_adap)
                this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                    perturbed_poi_list_for_dist[i + 1], reversed_perturbed_poi_post_direc, poi_angel_num_matrix_adap)

                this_poi_pre_direc_support_array = np.intersect1d(mask, this_poi_pre_direc_support_array).reshape(-1)

                this_poi_post_direc_support_array = np.intersect1d(mask, this_poi_post_direc_support_array).reshape(-1)

                this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                              this_poi_post_direc_support_array)
                # if len(this_poi_direc_support_array) == 0:
                    # print('this_poi_direc_support_array:',this_poi_direc_support_array)
            else:
                # 起点
                if i == 0:

                    perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i]],
                                                   poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                        get_reversed_num_direc_of_traj([perturbed_poi_post_direc],
                                                       direc_num=len(poi_angel_num_adap_list))[
                            0]
                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       reversed_perturbed_poi_post_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)

                # 终点
                elif i == len(poi_list_for_direc_dist) - 1:

                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[-1]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)

                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[-1],
                                                                                       perturbed_poi_pre_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)


                else:

                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)
                    perturbed_poi_post_direc = kRR(
                        poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i + 1]],
                        poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                        get_reversed_num_direc_of_traj([perturbed_poi_post_direc],
                                                       direc_num=len(poi_angel_num_adap_list))[
                            0]

                    this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i], perturbed_poi_pre_direc, poi_angel_num_matrix_adap)
                    this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i + 1], reversed_perturbed_poi_post_direc,
                        poi_angel_num_matrix_adap)

                    this_poi_pre_direc_support_array = np.intersect1d(mask, this_poi_pre_direc_support_array).reshape(
                        -1)

                    this_poi_post_direc_support_array = np.intersect1d(mask, this_poi_post_direc_support_array).reshape(
                        -1)

                    this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                                  this_poi_post_direc_support_array)

            this_poi_dist_support_array = poi_distance_matrix[this_poi][this_poi_direc_support_array]


            if len(this_poi_dist_support_array) > 0:
                if len(this_poi_direc_support_array) != 1:

                    delta_of_this_poi_direc = delta_poi_distance_matrix
                    over = np.exp(epsilon_per_poi * (- this_poi_dist_support_array / (2 * delta_of_this_poi_direc)))

                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = this_poi_direc_support_array[perturbed_poi_index_of_direc_support_array]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
                else:
                    perturbed_poi_index = this_poi_direc_support_array[0]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
            else:
                # print(perturbed_poi_list_for_dist[i - 1],perturbed_poi_pre_direc)
                x = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i - 1], perturbed_poi_pre_direc,
                                                        poi_angel_num_matrix_adap)
                # print('x:',x)
                y = np.intersect1d(mask, x).reshape(-1)
                # print('y:',y)
                if len(y > 0):

                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                            2 * delta_of_this_poi_direc))

                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
                else:
                    x = find_all_support_of_this_poi(this_poi, 1, poi_angel_num_matrix_adap)
                    y = np.intersect1d(mask, x).reshape(-1)
                    # print(x,y)

                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                            2 * delta_of_this_poi_direc))

                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
                perturbed_poi_str = index2str_dict[perturbed_poi_index]

            perturbed_poi_list_for_direc_dist.append(perturbed_poi_str)
        if kRR_exp_kRR == False:
            perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[0]])
            for i, poi_str in enumerate(perturbed_poi_list_for_direc_dist):
                perturbed_traj.append(poi_str)
                perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[i + 1]])
        else:
            perturbed_traj.append(perturbed_poi_list_for_direc_dist[0])
            for i, poi_idx in enumerate(perturbed_poi_list_for_dist):
                perturbed_traj.append(index2str_dict[poi_idx])
                perturbed_traj.append(perturbed_poi_list_for_direc_dist[i + 1])

    else:
        for i, this_poi in enumerate(poi_list_for_direc_dist):
            if kRR_exp_kRR == False:
                if i != len(poi_list_for_direc_dist) - 1:
                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)
                    perturbed_poi_post_direc = kRR(
                        poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i + 1]],
                        poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                        get_reversed_num_direc_of_traj([perturbed_poi_post_direc],
                                                       direc_num=len(poi_angel_num_adap_list))[
                            0]

                    this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i], perturbed_poi_pre_direc, poi_angel_num_matrix_adap)
                    this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i + 1], reversed_perturbed_poi_post_direc,
                        poi_angel_num_matrix_adap)

                    this_poi_pre_direc_support_array = np.intersect1d(mask, this_poi_pre_direc_support_array).reshape(
                        -1)

                    this_poi_post_direc_support_array = np.intersect1d(mask, this_poi_post_direc_support_array).reshape(
                        -1)

                    this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                                  this_poi_post_direc_support_array)
                else:
                    perturbed_poi_pre_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                  poi_angel_num_adap_list, epsilon_direction)

                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       perturbed_poi_pre_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)

            else:
                if i == 0:
                    perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i]][this_poi],
                                                   poi_angel_num_adap_list, epsilon_direction)

                    this_poi_direc_support_array = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i],
                                                                                       perturbed_poi_post_direc,
                                                                                       poi_angel_num_matrix_adap)

                    this_poi_direc_support_array = np.intersect1d(mask, this_poi_direc_support_array).reshape(-1)

                else:
                    perturbed_poi_pre_direc = kRR(
                        poi_angel_num_matrix_adap[perturbed_poi_list_for_dist[i - 1]][this_poi],
                        poi_angel_num_adap_list, epsilon_direction)
                    perturbed_poi_post_direc = kRR(poi_angel_num_matrix_adap[this_poi][perturbed_poi_list_for_dist[i]],
                                                   poi_angel_num_adap_list, epsilon_direction)

                    reversed_perturbed_poi_post_direc = \
                        get_reversed_num_direc_of_traj([perturbed_poi_post_direc],
                                                       direc_num=len(poi_angel_num_adap_list))[
                            0]

                    this_poi_pre_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i - 1], perturbed_poi_pre_direc, poi_angel_num_matrix_adap)
                    this_poi_post_direc_support_array = find_this_direc_support_of_this_poi(
                        perturbed_poi_list_for_dist[i], reversed_perturbed_poi_post_direc, poi_angel_num_matrix_adap)

                    this_poi_pre_direc_support_array = this_poi_pre_direc_support_array[mask].reshape(-1)

                    this_poi_post_direc_support_array = this_poi_post_direc_support_array[mask].reshape(-1)

                    this_poi_direc_support_array = np.intersect1d(this_poi_pre_direc_support_array,
                                                                  this_poi_post_direc_support_array)

            this_poi_dist_support_array = poi_distance_matrix[this_poi][this_poi_direc_support_array]

            if len(this_poi_dist_support_array) > 0:
                if len(this_poi_direc_support_array) != 1:
                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- this_poi_dist_support_array / (2 * delta_of_this_poi_direc)))
                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = this_poi_direc_support_array[perturbed_poi_index_of_direc_support_array]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
                else:
                    perturbed_poi_index = this_poi_direc_support_array[0]
                    perturbed_poi_str = index2str_dict[perturbed_poi_index]
            else:
                x = find_this_direc_support_of_this_poi(perturbed_poi_list_for_dist[i - 1], perturbed_poi_pre_direc,
                                                        poi_angel_num_matrix_adap)
                y = np.intersect1d(mask, x).reshape(-1)
                if len(y > 0):
                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                            2 * delta_of_this_poi_direc))
                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]
                else:
                    x = find_all_support_of_this_poi(this_poi, 1, poi_angel_num_matrix_adap)
                    y = np.intersect1d(mask, x).reshape(-1)
                    delta_of_this_poi_direc = delta_poi_distance_matrix

                    over = np.exp(epsilon_per_poi * (- poi_distance_matrix[this_poi][y]).reshape(-1) / (
                            2 * delta_of_this_poi_direc))
                    total = over.sum()

                    pdf = over / total
                    cdf = pdf.copy()
                    for j in range(1, pdf.shape[0]):
                        cdf[j] += cdf[j - 1]
                    rand_uni_num = np.random.random()

                    perturbed_poi_index_of_direc_support_array = np.searchsorted(cdf, rand_uni_num, side='right')
                    perturbed_poi_index = y[perturbed_poi_index_of_direc_support_array]

                perturbed_poi_str = index2str_dict[perturbed_poi_index]

            perturbed_poi_list_for_direc_dist.append(perturbed_poi_str)
        # 生成扰动轨迹
        if kRR_exp_kRR == False:
            for i, poi_str in enumerate(perturbed_poi_list_for_direc_dist):
                perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[i]])
                perturbed_traj.append(poi_str)
        else:
            for i, poi_str in enumerate(perturbed_poi_list_for_direc_dist):
                perturbed_traj.append(poi_str)
                perturbed_traj.append(index2str_dict[perturbed_poi_list_for_dist[i]])

    return perturbed_traj


def Exp_mechanism(traj, epsilon):
    traj_array = []
    for this_poi_str in traj:
        traj_array.append(str2index_dict[this_poi_str])
    traj_array = np.array(traj_array)
    delta = delta_poi_distance_matrix
    # epsilon 分配
    epsilon = epsilon / len(traj)
    epsilon_array = np.array(epsilon)
    over = np.exp(epsilon_array * (- poi_distance_matrix[traj_array]) / (2 * delta))
    total = over.sum(axis=1).reshape((-1, 1))
    cdf = over / total

    pdf = cdf.copy().T
    for j in range(1, cdf.shape[1]):
        pdf[j] += pdf[j - 1]
    pdf = pdf.T

    rand_uni_num = np.random.random((pdf.shape[0], 1))

    perturbed_traj = traj_array.copy()
    for i in range(rand_uni_num.shape[0]):
        perturbed_traj[i] = np.searchsorted(pdf[i], rand_uni_num[i], side='right')

    # 指数机制扰动轨迹结果
    perturbed_traj_str = []
    for this_perturbed_poi in perturbed_traj:
        perturbed_traj_str.append(index2str_dict[this_perturbed_poi])

    return perturbed_traj_str

def perturb_traj(traj, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list, mechanism='Exp_mechanism',
                 kRR_exp_kRR=False, traj_idx=None):
    if mechanism == 'ATP_mechanism':
        perturb_mechanism = ATP_mechanism
    else:
        print('please choose a correct mechanism.')

    perturbed_traj = perturb_mechanism(traj, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list, kRR_exp_kRR,
                                       traj_idx=traj_idx)

    errors_of_perturbed_traj = get_errors_of_perturbed_traj(traj, perturbed_traj)

    return perturbed_traj, errors_of_perturbed_traj


def find_opt_traj(perturbed_traj, perturbed_reversed_traj):
    opt_traj = []
    perturbed_reversed_traj = list(reversed(perturbed_reversed_traj))
    for i in range(len(perturbed_traj)):
        dist_matrix = poi_distance_matrix[str2index_dict[perturbed_traj[i]]] + \
                      poi_distance_matrix[str2index_dict[perturbed_reversed_traj[i]]]
        min_poi_idx = np.argmin(dist_matrix)
        opt_traj.append(index2str_dict[min_poi_idx])
    return opt_traj

# 真实轨迹点和噪音轨迹点的距离
def get_errors_of_perturbed_traj(traj, perturbed_traj):
    errors_of_perturbed_traj = 0

    for i, poi in enumerate(traj):
        this_poi_error = poi_distance_matrix[str2index_dict[poi]][str2index_dict[perturbed_traj[i]]]
        errors_of_perturbed_traj += this_poi_error
    normed_errors_of_perturbed_traj = errors_of_perturbed_traj / len(traj)
    return normed_errors_of_perturbed_traj

def get_preservation_range_queries_errors_of_perturbed_traj(traj, perturbed_traj):
    errors_of_perturbed_traj = 0

    # dist_list = [1,2,3,4]
    # For CPS:
    dist_list = [0.25]
    normed_errors_of_perturbed_traj = np.zeros((1, len(dist_list)))

    for dist_idx, dist in enumerate(dist_list):
        errors_of_perturbed_traj = 0
        for i, poi in enumerate(traj):
            this_poi_error = 0
            if poi_distance_matrix[str2index_dict[poi]][str2index_dict[perturbed_traj[i]]] <= dist:
                this_poi_error = 1
            errors_of_perturbed_traj += this_poi_error
        normed_errors_of_perturbed_traj[0][dist_idx] = errors_of_perturbed_traj / len(traj)
    return normed_errors_of_perturbed_traj


def get_NE(traj_list_without_timestamp,epsilon,poi_angel_num_matrix_adap,poi_angel_num_adap_list,threshold_value):
    NE = 0
    traj_domain = list(str2index_dict.keys())
    for r in range(0, 5):
        error = 0
        for i, trajectory in enumerate(traj_list_without_timestamp):
            for poi in trajectory:
                if poi not in traj_domain:
                    trajectory.remove(poi)
            trajectory = trajectory[0:threshold_value]
            reversed_trajectory = list(reversed(trajectory))
            perturbed_traj = ATP_mechanism(trajectory, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list,
                                           0.75,
                                           False, traj_idx=i)
            perturbed_reversed_traj, _ = perturb_traj(reversed_trajectory, epsilon / 2, poi_angel_num_matrix_adap,
                                                      poi_angel_num_adap_list, mechanism='ATP_mechanism',
                                                      kRR_exp_kRR=True, traj_idx=i)
            perturbed_traj = find_opt_traj(perturbed_traj, perturbed_reversed_traj)
            normed_errors_of_perturbed_traj = get_errors_of_perturbed_traj(trajectory, perturbed_traj)
            error += normed_errors_of_perturbed_traj
        mean_error = error / len(traj_list_without_timestamp)
        print(mean_error)
        NE += mean_error
    return NE/5

def get_PRQ(traj_list_without_timestamp,epsilon,poi_angel_num_matrix_adap,poi_angel_num_adap_list):
    PRQ = 0
    for r in range(0, 5):
        error = 0
        for i, trajectory in enumerate(traj_list_without_timestamp):
            reversed_trajectory = list(reversed(trajectory))
            perturbed_traj = ATP_mechanism(trajectory, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list,
                                           0.75,
                                           False, traj_idx=i)
            perturbed_reversed_traj, _ = perturb_traj(reversed_trajectory, epsilon / 2, poi_angel_num_matrix_adap,
                                                      poi_angel_num_adap_list, mechanism='ATP_mechanism',
                                                      kRR_exp_kRR=True, traj_idx=i)
            perturbed_traj = find_opt_traj(perturbed_traj, perturbed_reversed_traj)
            normed_errors_of_perturbed_traj = get_preservation_range_queries_errors_of_perturbed_traj(trajectory,
                                                                                                      perturbed_traj)
            error += normed_errors_of_perturbed_traj
        mean_error = error / len(traj_list_without_timestamp)
        print(mean_error)
        PRQ += mean_error
    return PRQ/5

def research_hotspot(traj_list_without_timestamp,POIs):
    point_users = defaultdict(set)
    for i, trajectory in enumerate(traj_list_without_timestamp):
        unique_points = set(trajectory)
        # 记录用户访问关系
        for point in unique_points:
            point_users[point].add(i)

    point_counts = {
        point: len(users)
        for point, users in point_users.items()
    }
    result = pd.DataFrame(
        [(point, count) for point, count in point_counts.items()],
        columns=['Location ID', 'user_count']
    )

    real_top = result.nlargest(POIs, 'user_count', keep="first")
    return real_top


def get_ACD(traj_list_without_timestamp,epsilon,poi_angel_num_matrix_adap,poi_angel_num_adap_list,POIs):
    real_top = research_hotspot(traj_list_without_timestamp,POIs)
    real_top_points = real_top['Location ID'].tolist()
    real_counts = real_top['user_count'].values.tolist()
    real_point_set = set(real_top_points)
    round_error = 0
    for r in range(0, 5):
        point_counts = defaultdict(int)
        for i, trajectory in enumerate(traj_list_without_timestamp):
            reversed_trajectory = list(reversed(trajectory))
            perturbed_traj = ATP_mechanism(trajectory, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list,
                                           0.75,
                                           False, traj_idx=i)
            perturbed_reversed_traj, _ = perturb_traj(reversed_trajectory, epsilon / 2, poi_angel_num_matrix_adap,
                                                      poi_angel_num_adap_list, mechanism='ATP_mechanism',
                                                      kRR_exp_kRR=True, traj_idx=i)
            perturbed_traj = find_opt_traj(perturbed_traj, perturbed_reversed_traj)
            matched_points = set(perturbed_traj) & real_point_set
            for point in matched_points:
                point_counts[point] += 1
        noise_top = pd.DataFrame(
                [(ID, point_counts.get(ID, 0)) for ID in real_top_points],
                columns=['Location ID', 'user_count']
            )
        noise_counts = noise_top['user_count'].values.tolist()
        error_sum = sum(abs(real - noise) for real, noise in zip(real_counts, noise_counts)) / len(real_counts)
        print(r, 'th round error is ', error_sum)
        round_error += error_sum
    ACD = round_error /5
    return ACD

def get_KT(traj_list_without_timestamp,epsilon,poi_angel_num_matrix_adap,poi_angel_num_adap_list,POIs):
    real_top = research_hotspot(traj_list_without_timestamp,POIs)
    real_top = real_top.reset_index(drop=True)
    real_index = list(real_top.index)
    real_top_points = real_top['Location ID'].tolist()
    real_point_set = set(real_top_points)
    round_error = 0
    for r in range(0, 5):
        point_counts = defaultdict(int)
        for i, trajectory in enumerate(traj_list_without_timestamp):
            reversed_trajectory = list(reversed(trajectory))
            perturbed_traj = ATP_mechanism(trajectory, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list,
                                           0.75,
                                           False, traj_idx=i)
            perturbed_reversed_traj, _ = perturb_traj(reversed_trajectory, epsilon / 2, poi_angel_num_matrix_adap,
                                                      poi_angel_num_adap_list, mechanism='ATP_mechanism',
                                                      kRR_exp_kRR=True, traj_idx=i)
            perturbed_traj = find_opt_traj(perturbed_traj, perturbed_reversed_traj)
            matched_points = set(perturbed_traj) & real_point_set
            for point in matched_points:
                point_counts[point] += 1
        noise_top = pd.DataFrame(
                [(ID, point_counts.get(ID, 0)) for ID in real_top_points],
                columns=['Location ID', 'user_count']
            )
        noise_top = noise_top.reset_index(drop=True)
        noise_top = noise_top.sort_values(by='user_count', ascending=False)
        noise_index = list(noise_top.index)
        # print(real_index, noise_index)
        tau, _ = kendalltau(real_index, noise_index)
        print(r, 'th round error is ', tau)
        round_error += tau
        KT = round_error / 5
    return KT


if __name__ == "__main__":

    traj_list_without_timestamp = []
    traj_list_timestamp = []

    for traj in traj_list:
        this_traj = []
        this_traj_timestamp = []
        for this_poi in traj:
            this_traj.append(this_poi[0])
        traj_list_without_timestamp.append(np.array(this_traj))

    epsilon_list = [4]
    for epsilon in epsilon_list:
        print(epsilon)
        if epsilon == 10:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_12
            poi_angel_num_adap_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        elif epsilon <= 2:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_2
            poi_angel_num_adap_list = [0, 1]
        elif epsilon == 4:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_4
            poi_angel_num_adap_list = [0, 1, 2, 3]
        else:
            poi_angel_num_matrix_adap = poi_angel_num_matrix_6
            poi_angel_num_adap_list = [0, 1, 2, 3, 4, 5]

        # POIs = 100
        # KT = get_KT(traj_list_without_timestamp,epsilon,poi_angel_num_matrix_adap,poi_angel_num_adap_list,POIs)
        # print('KT:', KT)
        threshold_value = 4
        NE = get_NE(traj_list_without_timestamp, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list,threshold_value)
        print('NE:', NE)


    # NE = get_NE(traj_list_without_timestamp, epsilon, poi_angel_num_matrix_adap, poi_angel_num_adap_list)
    # print('NE:', NE)
    # PRQ = get_PRQ(traj_list_without_timestamp,epsilon,poi_angel_num_matrix_adap,poi_angel_num_adap_list)
    # print('PRQ:', PRQ)
    # epsilon=4
    # if epsilon == 10:
    #     poi_angel_num_matrix_adap = poi_angel_num_matrix_12
    #     poi_angel_num_adap_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # elif epsilon <= 2:
    #     poi_angel_num_matrix_adap = poi_angel_num_matrix_2
    #     poi_angel_num_adap_list = [0, 1]
    # elif epsilon == 4:
    #     poi_angel_num_matrix_adap = poi_angel_num_matrix_4
    #     poi_angel_num_adap_list = [0, 1, 2, 3]
    # else:
    #     poi_angel_num_matrix_adap = poi_angel_num_matrix_6
    #     poi_angel_num_adap_list = [0, 1, 2, 3, 4, 5]

    # POI_list = [200, 400, 600, 800, 1000]
    # POI_list = [400, 800, 1200, 1600, 2000]
    # POI_list = [52, 104, 157, 209, 262]
    # for POIs in POI_list:
    #     print('POIs:', POIs)
    #     ACD = get_ACD(traj_list_without_timestamp,epsilon,poi_angel_num_matrix_adap,poi_angel_num_adap_list, POIs)
    #     print('The ACD error:',ACD)
