# cdpath21-gan

![Network diagram](http://jcboyd.github.io/assets/cdpath21-gan/network_diagram.png)

Code to train self-supervised adversarial autoencoder for visual field expansion of histopathology tiles [[1]](#Reference).

* ```main.py``` contains training code.
* ```src/models.py``` defines GAN generator and discriminators.
* ```src/utils.py``` defines utility functions for training and graphing.
* ```config/``` defines ```.yml``` configuration files to set experiment parameters. 

## Installation

The Anaconda environment is specified in ```environment.yml```. The environment can be recreated using,

```
conda env create -f environment.yml
```

Tested with single NVIDIA P100 GPU, running Cuda 10.0.130, and PyTorch 1.9.0 with torchvision 0.10.0.

## Usage

```main.py``` is the training code, which requires two parameters
* ```job_number``` specifies a unique identifier for writing outputs
* ```config``` specifies configuration file path

See ```slurm_submit.sh``` for example.

## Config files

See [config/README.md](config/README.md) for a description of configuration options.

## Data

Experiments performed on [CRC](https://zenodo.org/record/1214456) and customised dataset from [CAMELYON17](https://camelyon17.grand-challenge.org/). The CAMELYON option is designed to be compatible with the [PatchCamelyon dataset](https://github.com/basveeling/pcam).

## Reference
[1] *Self-Supervised Representation Learning using Visual Field Expansion on Digital Pathology*, Joseph Boyd, Mykola Liashuha, Eric Deutsch, Nikos Paragios, Stergios Christodoulidis, Maria Vakalopoulou, CDpath ICCV 2021 [[PDF]](https://arxiv.org/abs/2109.03299)

### Reconstructions (CAMELYON17)

![Reconstructions](http://jcboyd.github.io/assets/cdpath21-gan/reconstructions.png)

### Samples (CAMELYON17, CRC)

![Samples](http://jcboyd.github.io/assets/cdpath21-gan/samples.png)
