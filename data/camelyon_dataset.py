import os
import h5py
import pandas as pd
from openslide import OpenSlide


slide_dir = './CAMELYON17/images'
df_metadata = pd.read_csv('./data/meta_split.csv')
wsis = {}

for set_name in ['train', 'valid', 'test']:

    df_set = df_metadata[df_metadata.set == set_name]
    nb_rows = len(df_set)
    x_set, y_set = np.empty((nb_rows, 256, 256, 3)), np.empty(nb_rows)

    for i, row in df_set.iterrows():
        file_name = os.path.join(slide_dir, '%s_%s.tif' % (row.patient, row.node))

        if not file_name in wsis.keys():
            wsi = OpenSlide(file_name)
            wsis[file_name] = wsi

        x_set[i] = wsis[file_name].read_region((row.x, row.y), level=0, size=(256, 256))
        y_set[i] = 0 if row.type == 'negative' else 1

    with h5py.File('camelyonpatch_level_2_split_%s_x' % set_name, 'w') as f:
        f.create_dataset('x', data = x_set)

    with h5py.File('camelyonpatch_level_2_split_%s_y' % set_name, 'w') as f:
        f.create_dataset('y', data = y_set)

