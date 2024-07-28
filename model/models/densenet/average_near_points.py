def average_near_points(pcl_batch, radius=0.01):
    total_count = 0
    total_points = 0
    
    for pcl_i in pcl_batch:
        nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(pcl_i)
        distances, indices = nbrs.radius_neighbors(pcl_i)
        
        count_near_points = [len(pts) for pts in indices]
        
        total_count += sum(count_near_points)
        total_points += len(count_near_points)
    
    average_count = total_count / total_points if total_points > 0 else 0
    return average_count
