def val(model, valid_generator, num_examples):
    model.eval()
    metric = torch.tensor([0.0])
    metric = metric.to('cuda')

    with torch.no_grad():
        for pcl1, pcl2 in valid_generator:
            pcl1 = pcl1.to('cuda').unsqueeze(0)
            pcl2 = pcl2.to('cuda').unsqueeze(0)
            pcl1 = pcl1.view(-1, 1024, 3)
            pcl2 = pcl2.view(-1, 2048, 3)
            out = model(pcl1)
            out = out.reshape(-1, 2048, 3)
            metric += torch.sum(kaolin.metrics.pointcloud.f_score(out, pcl2)) / num_examples

    return metric
