import torch
import kaolin
import numpy as np


def valid(model_enc, model_dec, valid_set, num_examples = 100):
    print("Valid....")
    elems = np.random.choice(len(valid_set), num_examples)
    metric = torch.tensor([0.0])
    metric = metric.to('cuda')
    
    for i in elems:
        data, images = valid_set[i]
        images = images.to('cuda')
        data = data.to('cuda')

        images = images.unsqueeze(0)
        out = model_enc(images)
        out = model_dec(out)
        out = out.reshape(-1, 2048, 3)

        data = data.unsqueeze(0)
        
        metric += kaolin.metrics.pointcloud.f_score(data, out) / num_examples
    
    return metric