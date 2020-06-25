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

## GAN
- parser description
  - -d : dataset name(mnist, fashio_manist or cifar10)
  - -b : batch size(default=128)
  - --epoch : the number of epochs(default=20)
  - --latentdim : latent dimention(default=100)
  - --leakyrelu : use leeaky-ReLU or not(default=True)
  - --tilt : tilt of leaky-ReLU(default=0.2)
  - --droprate : dropout rate(default=0.3)
  - --genoptim : generator's optimizer(default=adam)
  - --discoptim : discriminator's optimizer(default=adam)
  - --softlabel : use softlabel or not(default=False)

### Quick start
```
python3 ./train.py -d [dataset_name]
```
