import os
import pathlib

import h5py
import numpy as np
import torch
from PIL import Image

import fukuoka
from torch.utils.data import Dataset

from basic_utils import DataTypesFukuoka, Constants, get_params
from model_utils import get_data_transform
from depth_utils import colorized_surfnorm_fukuoka


def fukuoka_loader(path, data_type, params):
    if data_type == DataTypesFukuoka.Depth:
        results_dir = params.features_root + params.dataset + '/' + Constants.COLORIZED_DEPTH_SAVE + '/'
        if os.path.exists(results_dir):  # check whether the colorized depth data has been saved offline
            filename = path.split('/')[-1]
            filename = filename[:filename.find('.')]
            img_path = results_dir + filename + '.hdf5'
            img_file = h5py.File(img_path, 'r')
            return np.asarray(img_file[data_type])
        else:
            img = colorized_surfnorm_fukuoka(path)
            return np.asarray(img, dtype=np.float32)
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


def is_train_instance(ins_name):
    train_instances = fukuoka.get_train_instances()
    for train_ins in train_instances:
        if train_ins == ins_name:
            return True

    return False


class FukuokaRGBDScene(Dataset):

    def __init__(self, params, split, loader=None):
        self.params = params
        self.split = split
        self.loader = loader
        self.rgb_transform = get_data_transform(data_type=DataTypesFukuoka.RGB)
        self.depth_transform = get_data_transform(data_type=DataTypesFukuoka.Depth)
        self.data_prop = self._init_dataset()

    def __getitem__(self, index):
        rgb_file_path, depth_file_path = self.data_prop[index]

        rgb_datum = self.loader(rgb_file_path, DataTypesFukuoka.RGB, self.params)
        depth_datum = self.loader(depth_file_path, DataTypesFukuoka.Depth, self.params)

        rgb_datum = self.rgb_transform(rgb_datum)
        depth_datum = self.depth_transform(depth_datum)

        filename = rgb_file_path.split('/')[-1]
        cat_name = filename.split(Constants.DELIM)[0]
        label_id = int(fukuoka.class_name_to_id.get(cat_name))

        return rgb_datum, depth_datum, label_id, filename

    def __len__(self):
        return len(self.data_prop)

    def _init_dataset(self):
        data_prop = []
        data_path = os.path.join(self.params.dataset_path, 'eval-set/')     # the dataset is located under "eval-set/" directory

        for file in sorted(list(pathlib.Path(data_path).glob("*" + Constants.DELIM + DataTypesFukuoka.RGB + "*.png"))):
            temp = file.name.split(DataTypesFukuoka.RGB)
            file_id = temp[1][:-len('.png')]
            rgb_file_path = str(file)
            depth_file_path = data_path + temp[0] + DataTypesFukuoka.Depth + file_id + '.png'
            ins_name = temp[0].split(Constants.DELIM)[1]
            is_train = is_train_instance(ins_name)
            if self.split == 'eval':
                if not is_train:
                    data_prop.append((rgb_file_path, depth_file_path))

            else:
                if is_train:
                    data_prop.append((rgb_file_path, depth_file_path))

        return data_prop


