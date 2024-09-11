#Big readme update soon!

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

big update soon

### Dense model

big update soon

### Losses && Metrics

big update soon

### Dataset

We created our own [dataset](https://github.com/bananananacat/Generation-of-3D-Objects/blob/main/model/data/datasets.md) and used ShapeNet dataset.
There are some [scripts](https://github.com/bananananacat/Generation-of-3D-Objects/tree/main/model/data/data_collection) - render functions, functions to work with camera and lights, functions to create dataset, etc.
