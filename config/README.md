## Configuration options

### Experiment options

* ```data_dir``` : Specifies the path to the directory containing experiment data.
* ```tensorboard_dir``` : Specifies the path to the directory for writing tensorboard outputs.
* ```outputs_dir``` : Specifies the path to the directory.
* ```ckpt``` : Specifies the path to a previously training model checkpoint for further training.

### Training options

* ```total_iterations``` : Total number of batches shown to each model. Training epochs and progressive growing phases will be adjusted in length accordingly.
* ```nb_batch``` : Number of training samples per batch.
* ```lambda_l1``` : Hyperparameter weight for reconstruction (L1) loss.

### Model options

* ```x_dim``` : Size of crops used as model inputs (taken from image centers).
* ```z_dim``` : Size of latent space vector.
