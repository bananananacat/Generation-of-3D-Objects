#DenseNet train
save_dir = 'model'
save_nums_dir = 'nums'

upsampling_factor = 2
input_size = 3

pcl_path = 'ShapeNet_pointclouds/ShapeNet_pointclouds'
json_path = 'splits/splits/train_models.json'

val_pcl_path = 'ShapeNet_pointclouds/ShapeNet_pointclouds'
val_json_path = 'splits/splits/val_models.json'

params_train = {'batch_size': 32, 'shuffle': True}
params_val = {'batch_size': 64, 'shuffle': False}

training_set = BatchGen_DenceNet(json_path, pcl_path)
valid_set = BatchGen_DenceNet(val_json_path, val_pcl_path)
training_generator = torch.utils.data.DataLoader(training_set, **params_train)
valid_generator = torch.utils.data.DataLoader(valid_set, **params_val)

densenet = DenseNet(input_size, upsampling_factor)
densenet = densenet.to('cuda')

optim = torch.optim.Adam(densenet.parameters(), lr=1e-4)
points_loss_val = 0.001
counter_ep = 1
counter = 0

loss_list = []
val_list = []
loss_dict = {}
val_dict = {}

for i in range(12):
    print(f"{counter_ep} epoch")
    for data, target in training_generator:
        counter += 1
        print(counter)
        optim.zero_grad()
        data = data.to('cuda')
        target = target.to('cuda')
        out = densenet(data)
        out = out.reshape(-1, 2048, 3)
        target = target.view(-1, 2048, 3)
        #print(out)
        loss = kaolin.metrics.pointcloud.chamfer_distance(out, target)
        loss = torch.mean(loss) + points_loss_val * point_loss(out)
        if counter % 100 == 0:
            print(counter, loss.item())
            #loss_list.append(loss.item())
            loss_dict[counter] = loss.item()

        loss.backward()
        optim.step()

    score = val(densenet, valid_generator, len(valid_generator))
    print(f"epoch = {counter_ep}, score =", score.item())
    #val_list.append(score.item())
    val_dict[counter_ep] = score.item()
    torch.save(densenet.state_dict(), os.path.join(save_dir, f'densenet_{counter_ep}_epoch.pt'))

    densenet.train()
    counter_ep += 1

print("end of learning")
torch.save(densenet.state_dict(), 'densenet.pt')

with open(os.path.join(save_nums, 'loss'), 'wb') as f:
    pickle.dump(loss_dict, f)
with open(os.path.join(save_nums, 'val'), 'wb') as f:
    pickle.dump(loss_dict, f)
