#Big readme update soon! At this point, the paper with the code has been submitted to a computer vision conference.

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

### PreCERMIT model

This model generates point cloud with 1024 points from given arbitrary number of photos/renders from different views.

directory with model with all scripts: https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/models/PreCERMIT

### CERMIT model

This model generates point cloud with N*k points from given point cloud with N points(quality enchansment).

directory with model with all scripts: https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/models/CERMIT

### Losses && Metrics

update soon

### Dataset

We created our own [dataset](https://github.com/bananananacat/Generation-of-3D-Objects/blob/main/model/data/datasets.md) and used ShapeNet dataset.
There are some [scripts](https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/data/data_collection) - render functions, functions to work with camera and lights, functions to create dataset, etc.
