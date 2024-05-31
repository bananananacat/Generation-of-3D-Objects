# Generation-of-Point-Clouds
HSE course project (2nd year)


### Team:
- Lebedev Vasiliy
- Levin Mark
- Khammatov Nikita


### Overview
Our goal is a website with models, that generate a point cloud from given any number of photos from different angles. We used some ideas from this articles([latent matching for reconstruction of point cloud](https://arxiv.org/pdf/1807.07796); [dense reconstruction of point cloud](https://arxiv.org/pdf/1901.08906v1)) and relevant repos([latent matching](https://github.com/val-iisc/3d-lmnet/tree/master); [dense](https://github.com/val-iisc/densepcr/tree/master)).


### Installing Requarements

```console
foo@bar:~$ pip3 install -r requirements.txt
```

### Installing kaolin lib:
(also [here](https://kaolin.readthedocs.io/en/latest/index.html) is documentation)
```console
foo@bar:~$ pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
foo@bar:~$ pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu121.html
```

### Latent matching model

Generate point clouds from n images. Firstly, Autoencoder: from point cloud to same point cloud. This part is using 1D convolutional
Encoder and fully-connected Decoder. After that - image convolutional Encoder, latent matching Autoencoder and image 
Encoder. 

This Model, Encoders, Decoders, Batch loader and train script are locating in [here](https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/models/v2_generation)

### Dense model

This model is working with quality of point clouds: for example, for given 1024 points it will return you k * 1024 points. It has 2 different custom Encoders and one Decoder. 

DenseNet passes given point cloud through first Encoder; it extracts point features and global features. Second Encoder - Radius Encoder - extracts local points features; for extracting local point features has been written a function that detects the nearest points in a circle of radius r - for every point in given point cloud(it extracts 3 times - for radius = 0.03, 0.04, 0.05 - because all point clouds are normilized from 0 to 1 [here](https://github.com/bananananacat/Generation-of-3D-Objects/blob/main/model/models/densenet/utils/BatchGen_DenseNet.py)). Then, DenceNet concatenate all features and pass them through Decoder. Also, was written a function that generates point cloud of n * 2^k points.

This Model, Encoders, Decoders, functions and train script are locating in [here](https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/models/densenet).
Also, [here](https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/models/densenet/Generation_of_Point_Clouds.ipynb) is notebook with all DenseNet functions; you can put your dataset in your Google Drive and run all cells.

### Losses && Metrics

For loss we decided to use Chamfer Distance(you can import it from kaolin lib), because it suits the purpose of our task and is faster than, for example, EMD loss. F score was used as metric to evaluate quality of point clouds.

Also we combined Chamfer loss with custom point loss, which prevents points from accumulating in one place(you can see it in [here](https://github.com/bananananacat/Generation-of-3D-Objects/blob/main/model/models/v2_generation/utils/losses.py))

### Dataset

We created our own [dataset](https://github.com/bananananacat/Generation-of-3D-Objects/blob/main/model/data/datasets.md) and used ShapeNet dataset.
There are some [scripts](https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/data/data_collection) - render functions, functions to work with camera and lights, functions to create dataset, etc.
