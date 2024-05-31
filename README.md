# Generation-of-3D-Objects
HSE course project (2nd year)


### Team:
- Lebedev Vasiliy
- Levin Mark
- Khammatov Nikita


### Overview
Our goal is a website with models, that generate a point cloud from given any number of photos from different angles. We used some ideas from this articles([latent matching for reconstruction of point cloud](https://arxiv.org/pdf/1807.07796); [dense reconstruction of point cloud](https://arxiv.org/pdf/1901.08906v1)) and relevant repos([latent matching](https://github.com/val-iisc/3d-lmnet/tree/master); [dense](https://github.com/val-iisc/densepcr/tree/master)).

### #TODO NKH

### Dense model

This model is working with quality of point clouds: for example, for given 1024 points it will return you k * 1024 points. This model has 2 different custom Encoders and one Decoder. 
