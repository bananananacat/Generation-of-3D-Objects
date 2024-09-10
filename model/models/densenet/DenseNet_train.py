from importer import *
from model/models/densenet/utils/get_near_points import *
from model/models/densenet/utils/attention import *
from model/models/densenet/utils/Encoder import *
from model/models/densenet/utils/Decoder import *
from model/models/densenet/utils/DenseNet import *
from densenet_val import *
from model/models/densenet/utils/point_loss import *
from model/models/densenet/utils/BatchGen_DenseNet import *

if __name__ == "__main__":
    
    save_dir = '/model'
    save_nums_dir = '/nums'

    upsampling_factor = 2
    input_size = 3

    pcl_path = '/ShapeNet_pointclouds/ShapeNet_pointclouds'
    json_path = '/splits/splits/train_models.json'

    val_pcl_path = '/ShapeNet_pointclouds/ShapeNet_pointclouds'
    val_json_path = '/splits/splits/val_models.json'

    params_train = {'batch_size': 32, 'shuffle': True}
    params_val = {'batch_size': 32, 'shuffle': False}

    training_set = BatchGen_CERMIT(json_path, pcl_path)
    valid_set = BatchGen_CERMIT(val_json_path, val_pcl_path)
    training_generator = torch.utils.data.DataLoader(training_set, **params_train)
    valid_generator = torch.utils.data.DataLoader(valid_set, **params_val)

    cermit = CERMIT(input_size, upsampling_factor, num_near_points = 10)
    cermit = cermit.to('cuda')

    optim = torch.optim.Adam(qualitynet.parameters(), lr=1e-4)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optim,
    num_warmup_steps= 150, num_training_steps=len(training_generator), num_cycles=2)

    points_loss_val = 0.055
    counter_ep = 1
    counter = 0


    loss_dict = {}
    loss_val_dict = {}
    val_dict1 = {}
    val_dict2 = {}

    for i in range(20):
        print(f"{counter_ep} epoch \n")
        for data, target in training_generator:
            counter += 1
            #print(counter)
            optim.zero_grad()
            data = data.to('cuda')
            target = target.to('cuda')
            out = cermit(data)
            out = out.reshape(-1, 2048, 3)
            target = target.view(-1, 2048, 3)
            # print(out)
            loss = kaolin.metrics.pointcloud.chamfer_distance(out, target)
            ch = torch.mean(loss)
            plv = point_loss(out)
            loss = ch + points_loss_val * plv
            if counter % 100 == 0:
                print(counter, loss.item(), ch.item(), plv)
                if counter % 300 == 0:
                    current_lr = optim.param_groups[0]['lr']
                    print(f"Step: {counter}, LR: {current_lr}         ")
                loss_dict[counter] = loss.item()
                
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(cermit.parameters(), clip_value=1.0)
            optim.step()
            scheduler.step()
            
        score1, score2, loss_mean = val(qualitynet, valid_generator)
        print(f"epoch = {counter_ep}, score1 = {score1.item()}, score2 = {score2.item()}, loss_val_mean = {loss_mean.item()}")
        val_dict1[counter_ep] = score1.item()
        val_dict2[counter_ep] = score2.item()
        loss_val_dict[counter_ep] = loss_mean.item()
        torch.save(cermit.state_dict(), os.path.join(save_dir, f'cermit_{counter_ep}_epoch.pt'))

        qualitynet.train()
        counter_ep += 1
        
    print("end of learning")
    torch.save(cermit.state_dict(), 'cermit.pt')

    with open(os.path.join(save_nums_dir, 'loss'), 'wb') as f:
        pickle.dump(loss_dict, f)
    with open(os.path.join(save_nums_dir, 'val1'), 'wb') as f:
        pickle.dump(val_dict1, f)
    with open(os.path.join(save_nums_dir, 'val2'), 'wb') as f:
        pickle.dump(val_dict2, f)
    with open(os.path.join(save_nums_dir, 'loss_val'), 'wb') as f:
        pickle.dump(loss_val_dict, f)
