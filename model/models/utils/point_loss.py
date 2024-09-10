
def point_loss(points):
    loss = torch.empty(points.shape[0], dtype=torch.float)
    loss = loss.to('cuda')
    for i in range(points.shape[0]):
        dist = torch.cdist(points[i], points[i])
        dist = dist[(dist <= 0.01) & (dist >= 0.00001)]
        dist = 0.002 / dist
        loss[i] = dist.mean()
    loss = min(1.0, loss.mean() / 5.0)
    return loss
