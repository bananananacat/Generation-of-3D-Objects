# DenseNet train
# GPU is necessary

def point_loss(points):
    loss = torch.empty(points.shape[0], dtype=torch.float)
    loss = loss.to('cuda')
    for i in range(points.shape[0]):
        dist = torch.cdist(points[0], points[0])
        dist = dist[(dist <= 0.01) & (dist >= 0.00001)]
        dist = 0.002 / dist
        loss[i] = dist.mean()
    loss = min(1.0, loss.mean() / 5.0)
    return loss

batch_size = 32
upsampling_factor = 2
input_size = 3

pcl_path = '/content/extracted_files/ShapeNet_pointclouds'
json_path = '/content/extracted_files/splits/train_models.json'
params = {'batch_size': 32, 'shuffle': True}

training_set = BatchGen_DenceNet(json_path, pcl_path)
training_generator = torch.utils.data.DataLoader(training_set, **params)

densenet = DenseNet(input_size, upsampling_factor)
densenet = densenet.to('cuda')
mse = nn.MSELoss()

optim = torch.optim.Adam(densenet.parameters(), lr=1e-4)

counter = 0

points_loss_val = 0.001
pbar = tqdm(training_generator)
counter_ep = 1
counter = 0
for i in range(8):
    print(f"{counter_ep} epoch")
    for data, target in pbar:
        counter += 1
        optim.zero_grad()
        data = data.to('cuda')
        target = target.to('cuda')
        out = densenet(data)
        out = out.reshape(-1, 2048, 3)

        target = target.view(-1, 2048, 3)
      
        loss = kaolin.metrics.pointcloud.chamfer_distance(out, target) 
        loss = torch.mean(loss) + points_loss_val * point_loss(out)
      
        pbar.set_description(f"Loss: {loss.item()}")
        if counter % 250 == 0:
            print(counter, loss)
        loss.backward()
        optim.step()
      
    # saving model after every epoch
  
    base_path = '/content/other_drive/MyDrive/'
    file_name = f'densenet_{counter_ep}_epoch.pt'
    full_path = os.path.join(base_path, file_name)
    torch.save(densenet.state_dict(), full_path)
    counter_ep += 1
  
print("end of learning")
torch.save(densenet.state_dict(), 'densenet.pt')
