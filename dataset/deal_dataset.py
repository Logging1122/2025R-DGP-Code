
import pandas as pd
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform


# 将分类标签打在每个轨迹点上

def points_label(traj_points, df):

    # df_label = traj_points.drop_duplicates(subset=['Cluster_Label'])
    label = traj_points['Cluster_Label'].values.tolist()
    label_dict = {}
    for l in label:
        label_points = traj_points[traj_points['Cluster_Label'] == l]
        points = label_points[['Latitude', 'Longitude']].values.tolist()
        label_dict[l] = points

    UID = df['User ID'].unique()
    label = []
    for ID in UID:
        df_ID = df[df['User ID'] == ID]
        if len(df_ID) == 1:
            df = df.drop(df[df['User ID'] == ID].index)
        else:
            trajectory = df_ID[['Latitude', 'Longitude']].values.tolist()
            # print(trajectory)
            for point in trajectory:
                for key, values in label_dict.items():
                    if point in values:
                        l = key
                        label.append(l)
    df['Cluster_Label'] = label
    return df



def traj_list(data):
    UID = data['User ID'].unique()
    trajs_list = []
    for ID in UID:
        df_ID = data[data['User ID'] == ID]
        # traj_list = df_ID[['Venue ID', 'Timestamp']].values.tolist()
        # Venue ID Location ID
        if len(df_ID) > 1:
            trajs_list.append(len(df_ID))
    return trajs_list


def haversine_vectorized(points):
    """
    计算地理距离矩阵（单位：公里）
    :param points: 经纬度数组，形状 (n, 2)，[[lat1, lon1], [lat2, lon2], ...]
    :return: 距离矩阵 (n x n)
    """
    R = 6371  # 地球半径（公里）
    points = np.radians(points)  # 一次性转换为弧度

    # 利用广播机制生成所有点对的纬度差/经度差
    lat = points[:, 0]
    lon = points[:, 1]
    dLat = lat[:, None] - lat  # 形状 (n, n)
    dLon = lon[:, None] - lon

    # 向量化Haversine公式
    a = np.sin(dLat /2)**2 + np.cos(lat[:, None]) * np.cos(lat) *  np.sin(dLon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist_matrix = R * c

    return dist_matrix




if __name__ == "__main__":
       Path = 'E:\\Code\\Trace-2\\dataset\\CPS\\'
       df = pd.read_csv(Path + 'CPS.csv', encoding='unicode_escape')
       columns = df.columns.tolist()
       traj_points = pd.read_csv(Path + 'traj_points.csv', encoding='unicode_escape')
       points_domain = traj_points[['Latitude', 'Longitude']].values.tolist()

       trajs_list = traj_list(df)
       print(trajs_list)
       print(min(trajs_list),max(trajs_list))

