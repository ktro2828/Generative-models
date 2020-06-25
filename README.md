# Generative-models
Generative models using Keras and Pytorch

# Requirement
- python3
- keras or pytorch

# Keras
## VAE
- parser description
  - -d : dataset name(mnist or fashion_mnist)
  - --latentdim : latent dimention(default=2)
  - --epoch : the number of epochs(default=2)
  - -b : the number of batch size(default=16)

### Quick start
```
python3 ./VAE.py -d [dataset_name]
```
