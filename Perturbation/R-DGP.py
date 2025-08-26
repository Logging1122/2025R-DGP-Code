import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import math
import krr
import random
from collections import defaultdict
from scipy.stats import kendalltau

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = np.radians(lat2 - lat1)
    dLon = np.radians(lon2 - lon1)
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dLon / 2) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dist = R * c
    return dist


def lat_lon_to_cartesian(latitude, longitude, earth_radius=6317000):
    """
    Convert latitude and longitude to Cartesian coordinates in a 2D Euclidean space.
    Parameters:
    - latitude (float): Latitude in degrees.
    - longitude (float): Longitude in degrees.
    - earth_radius (float): Earth's radius in meters (default is 6317000 meters).
    Returns:
    - (float, float): Cartesian coordinates (x, y).
    """
    # Convert latitude and longitude from degrees to radians
    phi = math.radians(latitude)
    eta = math.radians(longitude)
    # Calculate x and y coordinates
    x = earth_radius * math.sin(math.radians(90) - phi) * math.cos(eta)
    y = earth_radius * math.sin(math.radians(90) - phi) * math.sin(eta)
    coordinates = [x,y]
    return coordinates

def points_angle_dis_martix (trajectory_data):
    '''
    Calculate distance between  point and  point.
    Calculate angle between  point and  point.
    Returns
    --dis_matrix : distance matrix
    --angle_matrix : angle matrix

    '''
    dis_matrix = squareform(pdist(trajectory_data, metric='euclidean'))
    points_coordinates = []
    for traj_point in trajectory_data:
        coordinates = lat_lon_to_cartesian(traj_point[0], traj_point[1], earth_radius=6317000)
        points_coordinates.append(coordinates)

    points_coordinates=np.array(points_coordinates)
    dx = points_coordinates[:, np.newaxis, 0] - points_coordinates[:, 0]  # 计算 x2 - x1
    dy = points_coordinates[:, np.newaxis, 1] - points_coordinates[:, 1]  # 计算 y2 - y1
    angle_matrix = np.mod(np.arctan2(dy, dx), 2 * math.pi)

    return dis_matrix,angle_matrix

def Laplace_mechanism(sensitivity,epsilon):

    # Laplace_noise = np.random.laplace(loc=0, scale=sensitivity/epsilon)

    noise_rng = np.random.RandomState()
    Laplace_noise = noise_rng.exponential(scale=sensitivity/epsilon, size=1)

    return Laplace_noise

def square_wave_mechanism( value, max_value,epsilon ):

    # angle_max = 2*math.pi
    t = value / max_value
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
    perturbed_t = perturbed_t * max_value

    return perturbed_t

def traj_max_distance(trajectory):
    if len(trajectory) >= 2:
        points = np.array(trajectory)
        diff = points[1 :]-points[:-1]
        distances = np.sqrt((diff**2).sum(axis=1))
        dmax= round (distances.max() , 8)
    else:
        dmax = 0

    return dmax

def traj_set_max_distance(df):

    distance = [ ]
    UID = df['User ID'].unique()
    for ID in UID:
        df_ID = df[df['User ID'] == ID]
        traj = df_ID[['Latitude', 'Longitude']].values.tolist()
        dmax = traj_max_distance(traj)
        distance.append(dmax)
    dis_max = np.max(distance)
    return dis_max

def traj_point_perturb_region(pre_point,current_point,epsilon,trajectory_data,angle_matrix,dis_matrix,radius,depth=0):
    """
    Calculate the Angle between the current track point and the previous track point.The perturbation locus point region is obtained
    Parameters:
    - radius: noise radius
    - angle: The Angle between track points
    -
    Returns
    -candidates : Perturbation locus point region
    """
    pre_point_coord = lat_lon_to_cartesian(pre_point[0], pre_point[1], earth_radius=6317000)
    current_point_coord = lat_lon_to_cartesian(current_point[0], current_point[1], earth_radius=6317000)

    dx = current_point_coord[0] - pre_point_coord[0]
    dy = current_point_coord[1] - pre_point_coord[1]
    if np.arctan2(dy, dx) < 0:
        angle = np.mod(np.arctan2(dy, dx), 2 * math.pi)
    else:
        angle = np.arctan2(dy, dx)
    # add noise to the angle
    noise_angle = square_wave_mechanism(angle, 2 * math.pi, epsilon)

    center_idx = trajectory_data.index(current_point)

    radius_candidates = []
    points_dis = dis_matrix[center_idx]
    for idx, val in enumerate(points_dis):
        if val <= radius:
            radius_candidates.append(trajectory_data[idx])

    angle_candidates = []
    points_angle = angle_matrix[center_idx]
    for idx, val in enumerate(points_angle):
        if val <= noise_angle:
            angle_candidates.append(trajectory_data[idx])

    candidates = list(filter(lambda item: item in radius_candidates, angle_candidates))
    # candidates = list(filter(lambda item: item in trajectory_data, candidates))
    if current_point not in candidates:
        candidates.append(current_point)
    # if len(candidates) <= 1:
    #     return result_label[l]
    # else:
    #     return candidates
    return candidates

def Perturb_traj_points(trajectory,epsilon, trajectory_data, angle_matrix , dis_matrix,radius):
    """

    Parameters
    ----------
    trajectory  :  Individual user trajectories
    epsilon : privacy budget
    trajectory_data : List of trajectory points
    dis_matrix  : Trajectory points distance matrix
    angle_matrix : Trajectory points angle matrix
    Returns
    -real_coords , perturb_coords ：Coordinates of the original trajectory and the disturbed trajectory

    """
    point_eps = 0.4*epsilon / len(trajectory)
    angle_eps = 0.5*epsilon / (len(trajectory)-1)

    perturb_traj = []
    # first point
    point = trajectory[0]
    KRR = krr.KRR(trajectory_data, point_eps)
    perturbed_point = KRR.encode(tuple(point))
    perturb_traj.append(perturbed_point)

    for i in range(1, len(trajectory)):
        point = trajectory[i]
        candidates = traj_point_perturb_region(perturb_traj[i-1], point, angle_eps, trajectory_data, angle_matrix ,dis_matrix,
                                               radius)
        KRR = krr.KRR(candidates, point_eps)
        perturbed_point = KRR.encode(tuple(point))
        perturb_traj.append(perturbed_point)

    return perturb_traj


def get_NE_traj_perurb(UID,points_domain, angle_matrix,dis_matrix , epsilon,dis_max):

    ID_distances = 0
    for ID in UID:
        haversine_v = np.vectorize(haversine)
        df_ID = df[df['User ID'] == ID]
        df_ID = df_ID.reset_index()
        # for i in range(len(trajectory)):
        #     if trajectory[i] not in points_domain:
        #         df_ID = df_ID.drop(index=i)
        trajectory = df_ID[['Latitude', 'Longitude']].values.tolist()
        real_lat = list(map(lambda x: x[0], trajectory))
        real_lon = list(map(lambda x: x[1], trajectory))
        # if len(trajectory) > threshold_value:
        #     trajectory = trajectory[0:threshold_value]
        #     real_lat = list(map(lambda x: x[0], trajectory))
        #     real_lon = list(map(lambda x: x[1], trajectory))
        # else:
        #     real_lat = list(map(lambda x: x[0], trajectory))
        #     real_lon = list(map(lambda x: x[1], trajectory))

        # adjacent points distance
        traj_dis_max = traj_max_distance(trajectory)
        radius_eps = 0.1 * epsilon
        radius = square_wave_mechanism(traj_dis_max, dis_max, radius_eps)
        perturb_traj = Perturb_traj_points(trajectory, epsilon, points_domain, angle_matrix, dis_matrix, radius)
        noise_lat = list(map(lambda x: x[0], perturb_traj))
        noise_lon = list(map(lambda x: x[1], perturb_traj))

        distances = haversine_v(real_lat, real_lon, noise_lat, noise_lon)
        ID_distances += (sum(distances) / len(trajectory))
    return ID_distances


def Normalized_Error(run_round, traj_points, df, epsilon):

    points_domain = traj_points[['Latitude','Longitude']].values.tolist()
    dis_matrix, angle_matrix = points_angle_dis_martix(points_domain)
    # print(dis_matrix)
    # the maximum distance between adjacent points all tracks
    dis_max = traj_set_max_distance(df)
    UID = df['User ID'].unique()
    round_error = 0
    for r in range(0, run_round):
        # ID_distances = get_NE_traj_perurb(UID,points_domain, angle_matrix, dis_matrix , epsilon,dis_max,threshold_value)
        ID_distances = get_NE_traj_perurb(UID, points_domain, angle_matrix, dis_matrix, epsilon, dis_max)
        each_round = ID_distances / len(UID)
        print(r, 'th round error is ', each_round)
        round_error += each_round
    NE = round_error / run_round
    return NE


def get_PRQ_traj_perturb(UID,points_domain, angle_matrix,dis_matrix , epsilon,dis_max,dist):

    norm_error = 0
    for ID in UID:
        haversine_v = np.vectorize(haversine)
        ID_distances = this_poi = 0
        df_ID = df[df['User ID'] == ID]
        trajectory = df_ID[['Latitude', 'Longitude']].values.tolist()
        real_lat = list(map(lambda x: x[0], trajectory))
        real_lon = list(map(lambda x: x[1], trajectory))

        traj_dis_max = traj_max_distance(trajectory)
        radius_eps = 0.1 * epsilon
        radius = square_wave_mechanism(traj_dis_max, dis_max, radius_eps)

        perturb_traj = Perturb_traj_points(trajectory, epsilon, points_domain, angle_matrix, dis_matrix, radius)

        noise_lat = list(map(lambda x: x[0], perturb_traj))
        noise_lon = list(map(lambda x: x[1], perturb_traj))

        distances = haversine_v(real_lat, real_lon, noise_lat, noise_lon)
        for dis in distances:
            if dis <= dist:
                this_poi = 1
            ID_distances += this_poi
        norm_error += ID_distances / len(trajectory)
    return norm_error


def Preservation_range_query_error(run_round,traj_points, df, epsilon):

    points_domain = traj_points[['Latitude', 'Longitude']].values.tolist()
    dis_matrix, angle_matrix = points_angle_dis_martix(points_domain)
    # dis_max = dis_matrix.max()
    dis_max = traj_set_max_distance(df)
    UID = df['User ID'].unique()
    dists = [0.25, 0.5, 0.75, 1]
    PRQ = []
    for dist in dists:
        round_error = 0
        for r in range(0, run_round):
            norm_error = get_PRQ_traj_perturb(UID, points_domain, angle_matrix, dis_matrix, epsilon, dis_max, dist)
            each_round = norm_error / len(UID)
            print(r, 'th round error is ', each_round)
            round_error += each_round
        prq = round_error / run_round
        PRQ.append(prq)
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

def get_ACD_traj_perturb(UID, points_domain, angle_matrix, dis_matrix, epsilon, dis_max,real_top):

        real_top_points = real_top[['Latitude', 'Longitude']].round(6).apply(tuple, axis=1).tolist()
        real_point_set = set(real_top_points)
        real_counts = real_top['user_count'].values.tolist()
        point_counts = defaultdict(int)
        for ID in UID:
            df_ID = df[df['User ID'] == ID]
            trajectory = df_ID[['Latitude', 'Longitude']].values.tolist()
            traj_dis_max = traj_max_distance(trajectory)
            radius_eps = 0.1 * epsilon
            radius = square_wave_mechanism(traj_dis_max, dis_max, radius_eps)

            perturb_traj = Perturb_traj_points(trajectory, epsilon, points_domain, angle_matrix, dis_matrix, radius
                                               )
            perturb_traj = [(round(p[0], 6), round(p[1], 6)) for p in perturb_traj]
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
        error_sum = error_sum/len(real_counts)

        return error_sum

def Average_count_distance( traj_points, df, epsilon,run_round,POIs):

    points_domain = traj_points[['Latitude', 'Longitude']].values.tolist()
    dis_matrix, angle_matrix = points_angle_dis_martix(points_domain)
    dis_max = traj_set_max_distance(df)

    real_top = research_hotspot(df,POIs)
    # print(real_top)
    UID = df['User ID'].unique()
    round_error = 0
    for r in range(0, run_round):
        error_sum = get_ACD_traj_perturb(UID, points_domain, angle_matrix, dis_matrix, epsilon, dis_max,real_top)
        print(r, 'th round error is ', error_sum)
        round_error += error_sum
    ACD = round_error / run_round
    return ACD


def get_KT_traj_perturb(UID, points_domain, angle_matrix, dis_matrix, epsilon, dis_max,real_top):
    real_top_points = real_top[['Latitude', 'Longitude']].round(6).apply(tuple, axis=1).tolist()
    real_point_set = set(real_top_points)
    real_index = list(real_top.index)
    point_counts = defaultdict(int)
    for ID in UID:
        df_ID = df[df['User ID'] == ID]
        trajectory = df_ID[['Latitude', 'Longitude']].values.tolist()
        traj_dis_max = traj_max_distance(trajectory)
        radius_eps = 0.1 * epsilon
        radius = square_wave_mechanism(traj_dis_max, dis_max, radius_eps)

        perturb_traj = Perturb_traj_points(trajectory, epsilon, points_domain, angle_matrix, dis_matrix, radius
                                           )
        perturb_traj = [(round(p[0], 6), round(p[1], 6)) for p in perturb_traj]
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
    dis_matrix, angle_matrix = points_angle_dis_martix(points_domain)
    dis_max = traj_set_max_distance(df)

    real_top = research_hotspot(df, POIs)
    real_top = real_top.reset_index(drop=True)
    UID = df['User ID'].unique()
    round_error = 0
    for r in range(0, run_round):
        error = get_KT_traj_perturb(UID, points_domain, angle_matrix, dis_matrix, epsilon,
                                  dis_max, real_top)
        print(r, 'th round error is ', error)
        round_error += error
    KT = round_error / run_round
    return KT


# Example usage
if __name__ == "__main__":
       Path = 'E:\\Code\\Trace-2\\dataset\\CLE\\'
       df = pd.read_csv(Path + 'CLE_new.csv', encoding='unicode_escape')
       traj_points = pd.read_csv(Path + 'traj_points.csv', encoding='unicode_escape')

       # run_round = 10
       # epsilon_list = [1,2, 3, 4]
       # POIs = 200
       # for epsilon in epsilon_list:
       #     print(epsilon)
       #     KT = Kendall_Tau( traj_points, df, epsilon,run_round,POIs)
       #     print('The Kendall Tau ', KT)

       # Aa= [78, 131, 144, 196]
       # for A in Aa:
       #     traj_points = traj_points[0:A]
       #     NE = Normalized_Error(5, traj_points, df, 4)
       #     print(NE)

       epsilon = [1,2,4,8,10]
       for i in epsilon:
           NE = Normalized_Error(5, traj_points, df, i)
           print("epsilon:",i,NE)
       #
       #     PRQ = Preservation_range_query_error(5, traj_points, df, i)
       #     print("epsilon:", i, PRQ)

       # POI_list = [200, 400, 600, 800, 1000]
       # POI_list = [400, 800, 1200, 1600, 2000]
       # POI_list = [52, 104, 157, 209, 262]
       # for POI in POI_list:
       #     ACD = Average_count_distance(traj_points, df, 4, 5, POI)
       #     print(ACD)

