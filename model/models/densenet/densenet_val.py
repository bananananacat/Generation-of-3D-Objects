def val(model, valid_generator):
    model.eval()
    metric1 = torch.tensor([0.0])
    metric1 = metric1.to('cuda')
    metric2 = torch.tensor([0.0])
    metric2 = metric2.to('cuda')
    num_examples = 0
    loss_mean = torch.tensor([0.0])
    loss_mean = loss_mean.to('cuda')
    with torch.no_grad():
        for pcl1, pcl2 in valid_generator:
            model.zero_grad()
            pcl1 = pcl1.to('cuda').unsqueeze(0)
            pcl2 = pcl2.to('cuda').unsqueeze(0)
            pcl1 = pcl1.view(-1, 1024, 3)
            pcl2 = pcl2.view(-1, 2048, 3)
            out = model(pcl1)
            loss_mean += torch.sum(kaolin.metrics.pointcloud.chamfer_distance(pcl1, pcl2))
            out = out.reshape(-1, 2048, 3)
            metric1 += torch.sum(kaolin.metrics.pointcloud.f_score(out, pcl2))
            metric2 += torch.sum(kaolin.metrics.pointcloud.f_score(out, pcl2, radius = 0.02))
            num_examples += out.shape[0]
            print(loss_mean)
    loss_mean /= num_examples
    metric1 /= num_examples
    metric2 /= num_examples
    return metric1, metric2, loss_mean
