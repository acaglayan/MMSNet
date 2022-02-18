import os
import pickle
import shutil

import numpy as np
import scipy.io as io

from basic_utils import get_params, Constants
from sunrgbd import load_props


def organize_dataset(params):
    split_file = os.path.join(params.dataset_path, 'allsplit.mat')
    sunrgbd_meta_file = os.path.join(params.dataset_path, 'SUNRGBDMeta.mat')
    sunrgbd_meta = io.loadmat(sunrgbd_meta_file)['SUNRGBDMeta']

    train_data_paths = io.loadmat(split_file)['alltrain'][0]
    eval_data_paths = io.loadmat(split_file)['alltest'][0]

    data_paths = {'train': train_data_paths, 'eval': eval_data_paths}

    for split in ["train", "eval"]:
        results_dir = params.dataset_path + 'eval-set/' + split
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for path in data_paths[split]:
            rgb_img, depth_img = load_props(params, str(path), split)

            if rgb_img.is_scene_challenge_category() and depth_img.is_scene_challenge_category():
                sequence = 'SUNRGBD' + rgb_img.sequence_name

                if sequence[-1] == '/':
                    sequence = sequence[:-1]

                depth_img.Rtilt = np.asarray(sunrgbd_meta[sunrgbd_meta['sequenceName'] == sequence]['Rtilt'][0],
                                             dtype=np.float32)
                depth_img.K = np.asarray(sunrgbd_meta[sunrgbd_meta['sequenceName'] == sequence]['K'][0],
                                         dtype=np.float32)

                file_full_id = rgb_img.label + Constants.DELIM + rgb_img.sequence_name[1:].replace('/', '_') + '_' + \
                               rgb_img.img_name.split('.jpg')[0]
                rgb_fullname = file_full_id + '.jpg'
                depth_fullname = file_full_id + '.png'

                rgb_result_filename = results_dir + '/' + rgb_fullname
                depth_result_filename = results_dir + '/' + depth_fullname
                cam_params_file = results_dir + '/' + file_full_id + '.pkl'

                shutil.copy(rgb_img.path, rgb_result_filename)
                shutil.copy(depth_img.path, depth_result_filename)

                depth_img.path = depth_result_filename  # update path with the new path
                depth_img.sequence_name = file_full_id
                with open(cam_params_file, 'wb') as f:
                    pickle.dump(depth_img, f, -1)
                    f.close()


if __name__ == '__main__':
    param = get_params()
    organize_dataset(param)
