import numpy as np
import pandas as pd
import krr
from collections import defaultdict
from scipy.stats import kendalltau

# 数据点
def perturb_domain(df):
    df_domain = df.drop_duplicates(subset=['Latitude', 'Longitude'])
    coordinate_domain = df_domain[['Latitude', 'Longitude']].values.tolist()
    return coordinate_domain

def exponential_mechanism(point, epsilon, candidates,dis_matrix):
    """
    Apply the exponential mechanism to perturb trajectory points.
    Args:
        points (list of tuples): Original trajectory points [(x1, y1), (x2, y2), ...].
        epsilon (float): Privacy budget.
        sensitivity (float): Sensitivity of the scoring function (e.g., maximum distance between points).
    Returns:
        perturbed_points (list of tuples): Perturbed trajectory points.
    """
    index = candidates.index(point)
    dis_point = dis_matrix[index]
    sensitivity = max(dis_point)
    probabilities = np.exp(epsilon * np.array(dis_point) / (2 * sensitivity))
    probabilities /= np.sum(probabilities)
    perturbed_idx = np.random.choice(len(candidates), 1, p=probabilities)[0]
    perturbed_point = candidates[perturbed_idx]

    return perturbed_point


# 在整个数据集里直接对轨迹点进行扰动
def Perturb_data(df_ID,epsilon,coordinate_domain):
    Perturb_result = []
    epsilon = epsilon / len(df_ID)
    for i in range(0, len(df_ID)):
        coordinate = list(df_ID[['Latitude', 'Longitude']].iloc[i])
        KRR = krr.KRR(coordinate_domain, epsilon)
        perturb_data = KRR.encode(tuple(coordinate))
        Perturb_result.append(list(perturb_data))

    return Perturb_result

# Haversine 度量两个轨迹点的距离
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c
    return dist

def Normalized_Error(round,df,epsilon,traj_points,threshold_value):

    points_domain = traj_points[['Latitude', 'Longitude']].values.tolist()
    haversine_v=np.vectorize(haversine)
    UID = df['User ID'].unique()
    NE=0
    for r in range(0,round):
        ID_distances=0
        for ID in UID:
            df_ID = df[df['User ID'] == ID]
            if len(df_ID) > threshold_value:
                df_ID = df_ID.iloc[0:threshold_value]
            real_lat = df_ID['Latitude'].values.tolist()
            real_lon = df_ID['Longitude'].values.tolist()
            perturb_traj = Perturb_data(df_ID, epsilon, points_domain)
            noise_lat = list(map(lambda x: x[0], perturb_traj))
            noise_lon = list(map(lambda x: x[1], perturb_traj))

            distances = haversine_v(real_lat, real_lon, noise_lat, noise_lon)
            ID_distances += sum(distances) / len(df_ID)
        round_error = ID_distances / len(UID)
        print(r, 'th round error:', round_error)
        NE += round_error
    NE = NE / round
    print(round, 'rounds total error:', NE)

    return NE

def Preservation_range_query_error(round,df,epsilon,traj_points):

    dist = 1
    points_domain = traj_points[['Latitude', 'Longitude']].values.tolist()
    haversine_v=np.vectorize(haversine)
    UID = df['User ID'].unique()
    round_error = 0
    for r in range(0,round):
        norm_error = 0
        for ID in UID:
            ID_distances = 0
            this_poi = 0
            df_ID = df[df['User ID'] == ID]
            real_lat = df_ID['Latitude'].values.tolist()
            real_lon = df_ID['Longitude'].values.tolist()

            perturb_traj = Perturb_data(df_ID, epsilon, points_domain)
            noise_lat = list(map(lambda x: x[0], perturb_traj))
            noise_lon = list(map(lambda x: x[1], perturb_traj))
            distances = haversine_v(real_lat, real_lon, noise_lat, noise_lon)
            for dis in distances:
                if dis <= dist:
                    this_poi = 1
                ID_distances += this_poi
            norm_error += ID_distances / len(df_ID)
        each_round = norm_error / len(UID)
        print(r, 'th round error is ', each_round)
        round_error += each_round
    PRQ = round_error / round
    return PRQ

def research_hotspot(df,POIs):
    unique_points_per_user = df.groupby('User ID').apply(lambda x: set(zip(x['Latitude'], x['Longitude'])))
    point_counts = defaultdict(int)
    for user_id, points in unique_points_per_user.items():
        for point in points:
            point_counts[point] += 1
    result = pd.DataFrame([(lon, lat, count) for (lon, lat), count in point_counts.items()],columns=['Latitude', 'Longitude', 'user_count'])

    # 确保用户计数列是整数类型
    result['user_count'] = result['user_count'].astype(int)

    # 选择population最大的前3个，相同的值选择第一个
    real_top = result.nlargest(POIs, 'user_count', keep="first")

    return real_top

def get_ACD_traj_perturb(UID,points_domain,epsilon, real_top):

    real_top_points = real_top[['Latitude', 'Longitude']].round(6).apply(tuple, axis=1).tolist()
    real_point_set = set(real_top_points)
    real_counts = real_top['user_count'].values.tolist()
    point_counts = defaultdict(int)
    for ID in UID:
        df_ID = df[df['User ID'] == ID]
        perturb_traj = Perturb_data(df_ID, epsilon, points_domain)
        matched_points = set(map(tuple, perturb_traj)) & real_point_set
        for point in matched_points:
            point_counts[point] += 1
    noise_top = pd.DataFrame(
            [(lat, lon, point_counts.get((lat, lon), 0)) for lat, lon in real_top_points],
            columns=['Latitude', 'Longitude', 'user_count']
        )
    # print(noise_top)
    noise_counts = noise_top['user_count'].values.tolist()
    error_sum = sum(abs(real - noise) for real, noise in zip(real_counts, noise_counts))
    error_sum = error_sum / len(real_counts)
    return error_sum

def Average_count_distance( traj_points, df, epsilon,run_round,POIs):

    points_domain = traj_points[['Latitude', 'Longitude']].values.tolist()
    real_top = research_hotspot(df,POIs)
    # print(real_top)
    UID = df['User ID'].unique()
    round_error = 0
    for r in range(0, run_round):
        error_sum = get_ACD_traj_perturb(UID,points_domain,epsilon, real_top)
        print(r, 'th round error is ', error_sum)
        round_error += error_sum
    ACD = round_error / run_round
    return ACD


def get_KT_traj_perturb(UID, points_domain, epsilon,real_top):
    real_top_points = real_top[['Latitude', 'Longitude']].round(6).apply(tuple, axis=1).tolist()
    real_point_set = set(real_top_points)
    real_index = list(real_top.index)
    point_counts = defaultdict(int)
    for ID in UID:
        df_ID = df[df['User ID'] == ID]
        perturb_traj = Perturb_data(df_ID, epsilon, points_domain)
        # print(perturb_traj)
        matched_points = set(map(tuple, perturb_traj)) & real_point_set
        for point in matched_points:
            point_counts[point] += 1
    noise_top = pd.DataFrame(
        [(lat, lon, point_counts.get((lat, lon), 0)) for lat, lon in real_top_points],
        columns=['Latitude', 'Longitude', 'user_count']
    )
    noise_top = noise_top.reset_index(drop=True)
    noise_top = noise_top.sort_values(by='user_count', ascending=False)
    noise_index = list(noise_top.index)
    # print(real_index,noise_index)
    tau, _ = kendalltau(real_index, noise_index)
    return tau

def Kendall_Tau( traj_points, df, epsilon,run_round,POIs):

    points_domain = traj_points[['Latitude', 'Longitude']].values.tolist()
    real_top = research_hotspot(df, POIs)
    real_top = real_top.reset_index(drop=True)
    UID = df['User ID'].unique()
    round_error = 0
    for r in range(0, run_round):
        error = get_KT_traj_perturb(UID, points_domain, epsilon, real_top)
        print(r, 'th round error is ', error)
        round_error += error
    KT = round_error / run_round
    return KT

if __name__ == "__main__":
        Path = 'E:\\Code\\Trace-2\\dataset\\CLE\\'
        df = pd.read_csv(Path + 'CLE_new.csv', encoding='unicode_escape')
        traj_points = pd.read_csv(Path + 'traj_points.csv', encoding='unicode_escape')
        # run_round = 5
        # epsilon_list = [1, 2, 3]
        # for epsilon in epsilon_list:
        #     print(epsilon)
        #     KT = Kendall_Tau( traj_points, df, epsilon,run_round,200)
        #     print('The Kendall Tau ', KT)

        # epsilon = [0.1, 1, 2, 4, 8, 10]
        # for i in epsilon:
        #     NE = Normalized_Error(5,df,i,traj_points)
        #     print("epsilon:", i, NE)
        threshold_value_list = [4, 6, 8, 10, 12]
        for threshold_value in threshold_value_list:
            NE = Normalized_Error(5,df,4,traj_points,threshold_value)
            print("threshold_value:", threshold_value, NE)

        # epsilon = [1, 2, 4, 8, 10]
        # for i in epsilon:
        #      PRQ = Preservation_range_query_error(5,df,i,traj_points)
        #      print(PRQ)
