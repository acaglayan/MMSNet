import os

import h5py
import numpy as np
import scipy.io as io
import torch
from torch.utils.data import Dataset
from PIL import Image
import nyuv2
from basic_utils import Constants, get_params, DataTypesNYUV2
from depth_utils import colorized_surfnorm_nyu
from model_utils import get_data_transform


class NYUV2RGBDScene(Dataset):
    def __init__(self, params, split, loader=None):
        self.params = params
        self.split = split
        self.loader = loader
        self.rgb_transform = get_data_transform(data_type=DataTypesNYUV2.RGB)
        self.depth_transform = get_data_transform(data_type=DataTypesNYUV2.Depth)
        self.data_prop, self.rgb_data_table, self.depth_data_table = self.__init_dataset()

    def __len__(self):
        return len(self.data_prop)

    def __getitem__(self, index):
        f_name = self.data_prop[index]
        cat_name = f_name.split(Constants.DELIM)[0]
        label_id = int(nyuv2.class_name_to_id.get(cat_name))
        ind = int(f_name.split(Constants.DELIM)[-1])

        rgb_img = self.rgb_data_table[ind, ]
        depth_img = self.depth_data_table[ind, ]

        rgb_datum = np.transpose(rgb_img, [2, 1, 0])
        rgb_datum = Image.fromarray(rgb_datum)  # otherwise the transform gives error, asking for PIL image

        depth_results_dir = self.params.dataset_path + '/' + Constants.COLORIZED_DEPTH_SAVE + '/'
        if os.path.exists(depth_results_dir):  # check whether the depth data has been saved offline
            depth_img_path = depth_results_dir + cat_name + Constants.DELIM + DataTypesNYUV2.Depth + Constants.DELIM + str(ind) + '.hdf5'
            depth_img_file = h5py.File(depth_img_path, 'r')
            depth_datum = np.asarray(depth_img_file[DataTypesNYUV2.Depth])
        else:
            depth_img = colorized_surfnorm_nyu(depth_img)
            depth_datum = np.asarray(depth_img, dtype=np.float32)

        depth_datum = np.transpose(depth_datum, [1, 0, 2])

        rgb_datum = self.rgb_transform(rgb_datum)
        depth_datum = self.depth_transform(depth_datum)

        return rgb_datum, depth_datum, label_id, f_name

    def __init_dataset(self):
        data_prop = []
        data_path = os.path.join(self.params.dataset_path, 'nyu_depth_v2_labeled.mat')

        with h5py.File(data_path, 'r') as f:
            rgb_data_table = np.asarray(f["images"])
            depth_data_table = np.asarray(f["depths"])

        split_path = os.path.join(self.params.dataset_path, 'splits.mat')
        scene_type_path = os.path.join(self.params.dataset_path, 'sceneTypes.txt')

        # scene_types = f["sceneTypes"] we saved this as a text file and load from the text file

        scene_types = np.genfromtxt(scene_type_path, dtype='str')

        if self.split == "train":
            data_nds = np.squeeze(io.loadmat(split_path)["trainNdxs"])
        else:
            data_nds = np.squeeze(io.loadmat(split_path)["testNdxs"])

        for ind in data_nds:
            # we do "ind - 1" because it is a matlab file (with matlab indices start from 1)
            if nyuv2.class_name_to_id.get(scene_types[ind - 1]) is None:
                cat_label = "others"
            else:
                cat_label = scene_types[ind - 1]

            f_name = cat_label + Constants.DELIM + str(ind - 1)
            data_prop.append(f_name)

        return data_prop, rgb_data_table, depth_data_table


