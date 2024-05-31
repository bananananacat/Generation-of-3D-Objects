from scipy.spatial import KDTree
import numpy as np
points = data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def reduce_points_kd_tree(points, max_points, min_distance):
  tree = KDTree(points)
  reduced_points = []
  visited = np.zeros(points.shape[0], dtype=bool)
  for i in range(points.shape[0]):
        if not visited[i]:
            indices = tree.query_ball_point(points[i], min_distance)
            if len(indices) > 1:
                cluster_points = points[indices]
                representative_point = cluster_points.mean(axis=0)
                reduced_points.append(representative_point)
                visited[indices] = True
            else:
                reduced_points.append(points[i])
                visited[i] = True
                if len(reduced_points) >= max_points:
                  break
  return np.array(reduced_points)
max_points = 2048
min_distance = 0.05
reduced_points = reduce_points_kd_tree(points, max_points, min_distance)
print(f'Количество точек после уменьшения: {reduced_points.shape[0]}')

def plot_points(points, title):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
  ax.set_title(title)
  plt.show()
plot_points(points, 'Original Points')
plot_points(reduced_points, 'Reduced Points')
