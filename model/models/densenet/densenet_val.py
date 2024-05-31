def val(model, valid_set, num_examples = 100):
    model = model.to('cuda') 
    elems = np.random.choice(len(valid_set), num_examples)
    metric = torch.tensor([0.0])
    metric = metric.to('cuda')
    
    for i in elems:
        pcl1, pcl2 = valid_set[i]
        pcl1 = pcl1.to('cuda').unsqueeze(0)
        pcl2 = pcl2.to('cuda').unsqueeze(0)
        pcl1 = pcl1.view(-1, 1024, 3)
        pcl2 = pcl2.view(-1, 2048, 3)
        out = model(pcl1)
        out = out.reshape(-1, 2048, 3)
        metric += kaolin.metrics.pointcloud.f_score(out, pcl2) / num_examples
    
    return metric
