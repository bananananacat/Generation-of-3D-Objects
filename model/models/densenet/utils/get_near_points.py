# This function finds 'num_near_points' nearest points for a given point cloud. This will be used for extracting local point features.
# If there are less then 'num_near_points' points inside a circle of radius=radius, returned ndarray will be filled with [0, 0, 0];
# it will contain [0, 0, 0] * (num_near_points - num of points inside circle) times (for each points).

def get_nearest_points(pcl, radius=0.5, num_near_points=10):
    points = None
    for step, pcl_i in enumerate(pcl):
        nbrs = NearestNeighbors(n_neighbors=num_near_points, radius=radius, algorithm='auto').fit(pcl_i)
        distances, indices = nbrs.radius_neighbors(pcl_i)
        n = pcl_i.shape[0]
        nearest_points_all = np.zeros((n, num_near_points, 3))
        for i in range(n):
            nearest_points_within_radius = pcl_i[indices[i]]
            if len(nearest_points_within_radius) >= num_near_points:
                sorted_indices = np.argsort(distances[i])[:num_near_points]
                nearest_points_all[i] = nearest_points_within_radius[sorted_indices]
            else:
                nearest_points_all[i, :len(nearest_points_within_radius)] = nearest_points_within_radius

        nearest_points_all = np.expand_dims(nearest_points_all, axis=0)

        if step == 0:
            points = nearest_points_all
        else:
            points = np.concatenate((points, nearest_points_all), axis=0)

    return points

# function for concatenate given tensor of nearest points (from function above) with point cloud
def concatenate_with_pcl(pcl, nearest_points_all):
    nearest_points_all_torch = torch.tensor(nearest_points_all, dtype=torch.float32).to('cuda')
    nearest_points_all_flattened = nearest_points_all_torch.view(nearest_points_all_torch.shape[0], nearest_points_all_torch.shape[1], -1)
    result = torch.cat((pcl, nearest_points_all_flattened), dim=2)
    return result
