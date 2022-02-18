import os
import pathlib
import pickle

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import sunrgbd
from basic_utils import DataTypesSUNRGBD, Constants, get_params
from depth_utils import colorized_surfnorm_sunrgbd
from model_utils import get_data_transform


def sunrgbd_loader(path, data_type, params):
    if data_type == DataTypesSUNRGBD.Depth:
        temp = path.split('/')
        results_dir = params.features_root + params.dataset + '/' + Constants.COLORIZED_DEPTH_SAVE + '/' + temp[-2]
        if os.path.exists(results_dir):  # check whether the colorized depth data has been saved offline
            filename = temp[-2] + '/' + temp[-1]
            filename = filename[:-len('.png')]
            img_path = results_dir + filename + '.hdf5'
            img_file = h5py.File(img_path, 'r')
            return np.asarray(img_file[data_type], dtype=np.float32)
        else:
            cam_param_file = path[:-len('.png')] + '.pkl'
            with open(cam_param_file, 'rb') as f:
                depth_img = pickle.load(f)
                f.close()
            img = colorized_surfnorm_sunrgbd(depth_img)
            return np.asarray(img, dtype=np.float32)
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class SUNRGBDScene(Dataset):
    def __init__(self, params, split, loader=None):
        self.params = params
        self.split = split
        self.loader = loader
        self.rgb_transform = get_data_transform(data_type=DataTypesSUNRGBD.RGB)
        self.depth_transform = get_data_transform(data_type=DataTypesSUNRGBD.Depth)
        self.data_prop = self._init_dataset()

    def __getitem__(self, index):
        rgb_path, depth_path = self.data_prop[index]
        cat_name = rgb_path.split('/')[-1].split(Constants.DELIM)[0]
        label_id = np.int(sunrgbd.class_name_to_id[cat_name])

        rgb_datum = self.loader(rgb_path, DataTypesSUNRGBD.RGB, self.params)
        depth_datum = self.loader(depth_path, DataTypesSUNRGBD.Depth, self.params)

        rgb_datum = self.rgb_transform(rgb_datum)
        depth_datum = self.depth_transform(depth_datum)

        return rgb_datum, depth_datum, label_id, rgb_path

    def __len__(self):
        return len(self.data_prop)

    def _init_dataset(self):
        data_prop = []
        data_path = os.path.join(self.params.dataset_path, 'eval-set/')  # the dataset is organized under "eval-set/" directory
        data_path = data_path + self.split

        for file in sorted(list(pathlib.Path(data_path).glob("*.jpg"))):
            file_full_id = str(file).split(".jpg")[0]

            rgb_path = file_full_id + ".jpg"
            depth_path = file_full_id + ".png"

            data_prop.append((rgb_path, depth_path))

        return data_prop

