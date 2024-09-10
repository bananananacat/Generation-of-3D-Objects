if __name__ == "__main__":
    shapenet_id_to_category = {
        '02691156': 'airplane',
        '02828884': 'bench',
        '02933112': 'cabinet',
        '02958343': 'car',
        '03001627': 'chair',
        '03211117': 'monitor',
        '03636649': 'lamp',
        '03691459': 'speaker',
        '04090263': 'rifle',
        '04256520': 'sofa',
        '04379243': 'table',
        '04401088': 'telephone',
        '04530566': 'vessel'
    }

    shapenet_category_to_id = {
        'airplane': '02691156',
        'bench' : '02828884',
        'cabinet' : '02933112',
        'car' : '02958343',
        'chair' : '03001627',
        'lamp' : '03636649',
        'monitor' : '03211117',
        'rifle' : '04090263',
        'sofa' : '04256520',
        'speaker' : '03691459',
        'table' : '04379243',
        'telephone' : '04401088',
        'vessel' : '04530566'
    }

    save_dir = '/model'
    save_nums_dir = '/nums'

    upsampling_factor = 2
    input_size = 3

    pcl_path = '/ShapeNet_pointclouds/ShapeNet_pointclouds'
    json_path = '/splits/splits/train_models.json'

    img_path = "/home/aysurkov/Generation_of_pointclouds/ShapeNetRendering/ShapeNetRendering"

    val_pcl_path = '/ShapeNet_pointclouds/ShapeNet_pointclouds'
    val_json_path = '/splits/splits/val_models.json'

    params_train = {'batch_size': 4, 'shuffle': True}
    params_val = {'batch_size': 4, 'shuffle': False}

    training_generator = CustomDataLoader(json_path, img_directory=img_path, dots_directory=pcl_path, **params_train)
    valid_generator = CustomDataLoader(json_path, img_directory=img_path, dots_directory=pcl_path, **params_val)

    precermit = PreCERMIT()
    precermit = precermit.to('cuda')

    optim = torch.optim.Adam(precermit.parameters(), lr=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optim,
        num_warmup_steps= 150, num_training_steps=len(training_generator), num_cycles=2)

    points_loss_val = 0.01
    counter_ep = 1
    counter = 0

    loss_dict = {}
    loss_val_dict = {}
    val_dict1 = {}
    val_dict2 = {}

    for i in range(20):
        print(f"{counter_ep} epoch \n")
        for img, pcl in training_generator:
            counter += 1
            optim.zero_grad()
            img = img.to('cuda')
            pcl = pcl.to('cuda')
            print(img.shape, pcl.shape)
            out = precermit(img)
            out = out.reshape(-1, 1024, 3)
            pcl = pcl.view(-1, 1024, 3)
            loss = kaolin.metrics.pointcloud.chamfer_distance(out, pcl)
            ch = torch.mean(loss)
            plv = point_loss(out)
            loss = ch + points_loss_val * plv
            if counter % 100 == 0:
                print(counter, loss.item())
                if counter % 300 == 0:
                    current_lr = optim.param_groups[0]['lr']
                    print(f"Step: {counter}, LR: {current_lr}         ")
                loss_dict[counter] = loss.item()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(precermit.parameters(), clip_value=1.0)
            optim.step()
            scheduler.step()

        score1, score2, loss_mean = val(precermit, valid_generator)
        print(f"epoch = {counter_ep}, score1 = {score1.item()}, score2 = {score2.item()}, loss_val_mean = {loss_mean.item()}")
        val_dict1[counter_ep] = score1.item()
        val_dict2[counter_ep] = score2.item()
        loss_val_dict[counter_ep] = loss_mean.item()
        torch.save(precermit.state_dict(), os.path.join(save_dir, f'precermit_{counter_ep}_epoch.pt'))

        precermit.train()
        counter_ep += 1

    print("end of learning")
    torch.save(precermit.state_dict(), 'precermit.pt')

    with open(os.path.join(save_nums_dir, 'loss'), 'wb') as f:
        pickle.dump(loss_dict, f)
    with open(os.path.join(save_nums_dir, 'val1'), 'wb') as f:
        pickle.dump(val_dict1, f)
    with open(os.path.join(save_nums_dir, 'val2'), 'wb') as f:
        pickle.dump(val_dict2, f)
    with open(os.path.join(save_nums_dir, 'loss_val'), 'wb') as f:
        pickle.dump(loss_val_dict, f)
