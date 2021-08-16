import os
import sys
import h5py

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from functools import reduce
from skimage.transform import resize


# ------------------
#  HELPER FUNCTIONS
# ------------------

def write_flush(*text_args, stream=sys.stdout):
    stream.write(', '.join(map(str, text_args)) + '\n')
    stream.flush()
    return


def count_params(net):
    nb_params = 0
    for param in net.parameters():
        nb_params += reduce(lambda x, y : x * y, param.shape)
    return nb_params

# ------------------
#  CAMELYON DATASET
# ------------------

def load_data(root_dir):
    data = h5py.File(os.path.join(root_dir, './camelyonpatch_level_2_split_train_x.h5'), 'r')
    x_train = data['x'][()]
    data = h5py.File(os.path.join(root_dir,'./camelyonpatch_level_2_split_train_y.h5'), 'r')
    y_train = data['y'][()].squeeze()

    data = h5py.File(os.path.join(root_dir,'./camelyonpatch_level_2_split_valid_x.h5'), 'r')
    x_valid = data['x'][()]
    data = h5py.File(os.path.join(root_dir,'./camelyonpatch_level_2_split_valid_y.h5'), 'r')
    y_valid = data['y'][()].squeeze()

    data = h5py.File(os.path.join(root_dir,'./camelyonpatch_level_2_split_test_x.h5'), 'r')
    x_test = data['x'][()]
    data = h5py.File(os.path.join(root_dir,'./camelyonpatch_level_2_split_test_y.h5'), 'r')
    y_test = data['y'][()].squeeze()

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def data_generator(x_data, y_data, nb_batch=32):

    while True:

        # Pytorch randint trick (1000x faster than torch.randperm):
        x = torch.arange(x_data.shape[0])
        idx = x[torch.randint(0, x.shape[0], (nb_batch,))]

        x_batch = x_data[sorted(idx)]
        y_batch = y_data[sorted(idx)]
        y_batch = torch.Tensor(y_batch.squeeze()).long()
        # normalise
        x_batch = np.moveaxis(x_batch, 3, 1).copy()
        x_batch = torch.Tensor(x_batch)
        x_batch = x_batch / 127.5 - 1  # range [-1, 1]
        # data augment
        yield x_batch, y_batch

# -------------
#  CRC DATASET
# -------------

def infinite_data_loader(data_loader):
    while True:
        for x_batch, y_batch in data_loader:
            yield x_batch, y_batch

class CRCDataset(Dataset):

    def __init__(self, root_dir, transform=None, crop_size=112):

        self.root_dir = root_dir
        categories = os.listdir(self.root_dir)

        self.files = []
        self.classes = []

        for category in categories:
            cat_path = os.path.join(self.root_dir, category)
            file_names = os.listdir(cat_path)
            self.files.extend([os.path.join(cat_path, file_name)
                               for file_name in file_names])
            self.classes.extend(len(file_names) * [category])

        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file_path = self.files[idx]
        image_class = self.classes[idx]

        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)

        return image, image_class

def fade_in(x, size, alpha):
    x_high = F.interpolate(x, size=size, mode='bilinear',
                           align_corners=True)

    x_low = F.interpolate(x, size=size//2, mode='bilinear',
                          align_corners=True)
    x_low = F.interpolate(x_low, size=size, mode='nearest')

    return alpha * x_high + (1 - alpha) * x_low

def scale_generator(x_batch, y_batch, size, alpha, x_dim, rescale_size=224):

    x_batch = F.interpolate(x_batch, size=rescale_size, mode='bilinear',
                            align_corners=True)

    margin = (rescale_size - x_dim) // 2
    x_crop = x_batch[:, :, margin:rescale_size-margin, margin:rescale_size-margin]

    ## Hack for autoencoder:
    #x_target = fade_in(x_batch[:, :, margin:rescale_size-margin, margin:rescale_size-margin], size, alpha)
    x_target = fade_in(x_batch, size, alpha)  # alpha = 1 => no fade

    return x_crop, x_target, y_batch

# ---------------
#  VISUALISATION
# ---------------

def create_mosaique(x_batch, nrows, ncols):

    x_batch = x_batch.permute((0, 2, 3, 1)).squeeze()

    mosaique = torch.empty((0,))

    for i in range(nrows):
        row = torch.cat(list(x_batch[i * ncols:(i + 1) * ncols]), axis=1)
        mosaique = torch.cat([mosaique, row])

    return (127.5 * (mosaique + 1)).numpy().astype('uint8')

def plot_tiles(imgs, emb, grid_units=50, pad=1):

    # roughly 1000 x 1000 canvas
    cell_width = 1000 // grid_units
    s = grid_units * cell_width

    nb_imgs = imgs.shape[0]

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx].permute(1, 2, 0)                
                
                resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))

                y = j * cell_width
                x = i * cell_width

                canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = resized_tile
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)

    return canvas, img_idx_dict
