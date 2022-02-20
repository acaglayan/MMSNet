import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import psutil
import torch


class Constants:
    DELIM = "__"
    COLORIZED_DEPTH_SAVE = "colorized_depth_images"


class DataTypesSUNRGBD:
    RGB = 'sun_rgb'
    Depth = 'sun_depth'
    RGBD = 'sun_rgbd'
    ALL = [RGB, Depth, RGBD]


class DataTypesFukuoka:
    RGB = 'fukuoka_rgb'
    Depth = 'fukuoka_depth'
    RGBD = 'fukuoka_rgbd'
    ALL = [RGB, Depth, RGBD]


class DataTypesNYUV2:
    RGB = 'nyu_rgb'
    Depth = 'nyu_depth'
    RGBD = 'nyu_rgbd'
    ALL = [RGB, Depth, RGBD]


class PrForm:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    END_FORMAT = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def squeeze_dic_arrays(dic):
    for key in dic.keys():
        dic[key] = np.squeeze(dic[key])

    return dic


def convert_listvars_to_array(listvars):
    return np.concatenate([np.array(i) for i in listvars])


def dictionary_to_tensor(inputs):
    num_level = len(inputs)
    batch_size = np.array(inputs.get("layer1")).shape[0]
    input_size = np.array(inputs.get("layer1")).shape[-1]
    narray = np.zeros((num_level, batch_size, input_size), np.float32)
    i = 0

    for key in inputs.keys():
        narray[i, :, :] = inputs[key]
        i += 1

    return narray


def numpy2tensor(np_var, device=torch.device("cuda")):
    return torch.from_numpy(np_var.copy()).to(device)


def tensor2numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")


def init_logger(logfile_name, params):
    os.makedirs(os.path.dirname(logfile_name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s',
                        filename=logfile_name, filemode='w')
    logging.info('Running params: {}'.format(params))
    logging.info('----------\n')


def init_save_dirs(params):
    if params.dataset == "sunrgbd":
        parent_path = 'sunrgbd'
    elif params.dataset == "nyuv2":
        parent_path = 'nyuv2'
    else:
        parent_path = 'fukuoka'

    annex = ''
    # if params.debug_mode:
    #     annex += '[debug]'

    params.models_path += annex + '/'
    params.log_dir += annex + '/' + parent_path + '/'

    return params


def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", dest="dataset", default="sunrgbd", choices=["sunrgbd", "nyuv2", "fukuoka"],
                        help="SUN RGB-D Scene [sunrgbd], NYU V2 Scene [nyuv2], or Fukuoka RGB-D Indoor Scene [fukuoka]")
    parser.add_argument("--dataset-path", dest="dataset_path", default="/media/gsrt/144AAC7A4AAC59EE/alic_files/datasets/sunrgbd/",
                        # /media/gsrt/144AAC7A4AAC59EE/alic_files/datasets/sunrgbd/  /media/gsrt/144AAC7A4AAC59EE/alic_files/datasets/nyuv2/  /media/gsrt/144AAC7A4AAC59EE/alic_files/datasets/fukuoka/
                        help="Path to the data root")
    parser.add_argument("--models-path", dest="models_path", default="/media/gsrt/144AAC7A4AAC59EE/alic_files/mmsnet",
                        help="Root folder for CNN features to load/save")
    parser.add_argument("--num-rnn", dest="num_rnn", default=128, type=int, help="Number of RNN")
    parser.add_argument("--reuse-randoms", dest="reuse_randoms", default=1, choices=[0, 1], type=int,
                        help="Handles whether the random weights are gonna save/load or not")
    parser.add_argument("--log-dir", dest="log_dir", default="../logs", help="Log directory")
    parser.add_argument("--batch-size", dest="batch_size", default=16, type=int)
    params = parser.parse_args()

    return params
