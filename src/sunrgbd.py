import os

import numpy as np

from basic_utils import DataTypesSUNRGBD

class_id_to_name = {
    "0": "bathroom",
    "1": "bedroom",
    "2": "classroom",
    "3": "computer_room",
    "4": "conference_room",
    "5": "corridor",
    "6": "dining_area",
    "7": "dining_room",
    "8": "discussion_area",
    "9": "furniture_store",
    "10": "home_office",
    "11": "kitchen",
    "12": "lab",
    "13": "lecture_theatre",
    "14": "library",
    "15": "living_room",
    "16": "office",
    "17": "rest_space",
    "18": "study_space"
}
class_name_to_id = {v: k for k, v in class_id_to_name.items()}

class_names = set(class_id_to_name.values())


def get_class_ids(names):
    ids = []
    for name in names:
        _id = class_name_to_id[name]
        ids.append(_id)

    return np.asarray(ids, dtype=np.int)


def get_class_names(ids):
    names = []
    for _id in ids:
        _name = class_id_to_name[str(_id)]
        names.append(_name)

    return np.asarray(names)


def _is_category_available(cat_name):
    for cat in class_names:
        if cat == cat_name:
            return True
    return False


def load_props(params, path, split):
    end_ind = path.rfind('\\') - 1
    rel_seq_path = path.split('SUNRGBD')[1][:end_ind]
    data_path = os.path.join(params.dataset_path, 'public_set')
    instance_path = data_path + rel_seq_path

    label = np.loadtxt(os.path.join(instance_path, 'scene.txt'), dtype=str)

    rgb_img_dir_name = 'image/'
    depth_img_dir_name = 'depth/'

    rgb_img_name = os.listdir(os.path.join(instance_path, rgb_img_dir_name))[0]
    rgb_path = os.path.join(instance_path, rgb_img_dir_name+rgb_img_name)

    depth_img_name = os.listdir(os.path.join(instance_path, depth_img_dir_name))[0]
    depth_path = os.path.join(instance_path, depth_img_dir_name+depth_img_name)

    # intrinsics = np.loadtxt(os.path.join(instance_path, 'intrinsics.txt'), dtype=np.float32)
    # extrinsics = np.loadtxt(os.path.join(
    #     instance_path, 'extrinsics/' + os.listdir(os.path.join(instance_path, 'extrinsics/'))[0]), dtype=np.float32)

    sunrgbd_image = SunRGBDImage(DataTypesSUNRGBD.RGB, rgb_img_name, rgb_path, str(label), split)
    sunrgbd_image.sequence_name = rel_seq_path
    # sunrgbd_image.intrinsics = intrinsics
    # sunrgbd_image.extrinsics = extrinsics

    sunrgbd_depth = SunRGBDImage(DataTypesSUNRGBD.Depth, depth_img_name, depth_path, str(label), split)
    sunrgbd_depth.sequence_name = rel_seq_path
    # sunrgbd_depth.intrinsics = intrinsics
    # sunrgbd_depth.extrinsics = extrinsics

    return sunrgbd_image, sunrgbd_depth


class SunRGBDImage:
    def __init__(self, data_type, img_name, path, label, split):
        self._data_type = data_type
        self._img_name = img_name
        self._path = path
        self._label = label
        self._split = split
        self._sequence_name = None
        self._Rtilt = None
        self._K = None
        self._intrinsics = None
        self._extrinsics = None

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        self._data_type = data_type

    @property
    def img_name(self):
        return self._img_name

    @img_name.setter
    def img_name(self, img_name):
        self._img_name = img_name

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        self._split = split

    @property
    def sequence_name(self):
        return self._sequence_name

    @sequence_name.setter
    def sequence_name(self, sequence_name):
        self._sequence_name = sequence_name

    @property
    def Rtilt(self):
        return self._Rtilt

    @Rtilt.setter
    def Rtilt(self, Rtilt):
        self._Rtilt = Rtilt

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, K):
        self._K = K

    def is_scene_challenge_category(self):
        return _is_category_available(self.label)
