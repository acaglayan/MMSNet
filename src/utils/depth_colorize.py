import os
import pathlib
import pickle

import h5py
import numpy as np
from scipy import io

import nyuv2
from basic_utils import get_params, Constants, DataTypesFukuoka, DataTypesNYUV2, DataTypesSUNRGBD
from depth_utils import colorized_surfnorm_fukuoka, colorized_surfnorm_nyu, colorized_surfnorm_sunrgbd


def colorize_fukuoka(params, results_dir):
    data_path = os.path.join(params.dataset_path, 'eval-set/')
    suffix = '*' + DataTypesFukuoka.Depth + '*.png'
    for file in sorted(list(pathlib.Path(data_path).glob(suffix))):
        result_filename = results_dir + str(file)[str(file).rfind('/') + 1:str(file).rfind(".")] + '.hdf5'

        with h5py.File(result_filename, 'w') as f:
            f.create_dataset(DataTypesFukuoka.Depth, data=np.array(colorized_surfnorm_fukuoka(file), dtype=np.float32))
        f.close()


def colorize_nyuv2(params, results_dir):
    data_path = os.path.join(params.dataset_path, 'nyu_depth_v2_labeled.mat')

    with h5py.File(data_path, 'r') as f:
        data_table = np.asarray(f["depths"])

    split_path = os.path.join(params.dataset_path, 'splits.mat')
    scene_type_path = os.path.join(params.dataset_path,
                                   'sceneTypes.txt')  # scene_types = f["sceneTypes"] we saved this as a text file and load from the text file

    scene_types = np.genfromtxt(scene_type_path, dtype='str')

    train_data_nds = np.squeeze(io.loadmat(split_path)["trainNdxs"])
    test_data_nds = np.squeeze(io.loadmat(split_path)["testNdxs"])
    data_nds = np.concatenate((test_data_nds, train_data_nds))

    for ind in data_nds:  # indices start from 1 (matlab), so we handle them accordingly
        if nyuv2.class_name_to_id.get(scene_types[ind - 1]) is None:
            cat_label = "others"
        else:
            cat_label = scene_types[ind - 1]

        f_name = cat_label + Constants.DELIM + DataTypesNYUV2.Depth + Constants.DELIM + str(ind - 1)
        result_filename = results_dir + f_name + '.hdf5'
        file = data_table[ind - 1]

        with h5py.File(result_filename, 'w') as f:
            f.create_dataset(DataTypesNYUV2.Depth, data=np.array(colorized_surfnorm_nyu(file), dtype=np.float32))
        f.close()


def colorize_sunrgbd(params, results_dir):
    data_path = os.path.join(params.dataset_path, 'eval-set/')
    suffix = '*.pkl'
    root_path = results_dir
    for split in ["eval", "train"]:
        results_dir = root_path + split + '/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for file in sorted(list(pathlib.Path(data_path + split).glob(suffix))):
            cam_params_file = str(file)
            with open(cam_params_file, 'rb') as f:
                depth_img = pickle.load(f)
                f.close()

            result_filename = results_dir + depth_img.sequence_name + '.hdf5'
            with h5py.File(result_filename, 'w') as f:
                f.create_dataset(DataTypesSUNRGBD.Depth, data=np.array(colorized_surfnorm_sunrgbd(depth_img), dtype=np.float32))
            f.close()


def colorize_depth_sets(params):

    results_dir = params.features_root + '/' + params.dataset + '/' + Constants.COLORIZED_DEPTH_SAVE + '/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if params.dataset == "fukuoka":
        colorize_fukuoka(params, results_dir)

    elif params.dataset == "nyuv2":
        colorize_nyuv2(params, results_dir)

    else:
        colorize_sunrgbd(params, results_dir)


if __name__ == '__main__':
    param = get_params()
    colorize_depth_sets(param)
