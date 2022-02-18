import numpy as np

class_id_to_name = {
    "0": "corridor",
    "1": "kitchen",
    "2": "lab",
    "3": "office",
    "4": "study_room",
    "5": "toilet"
}
class_name_to_id = {v: k for k, v in class_id_to_name.items()}

class_names = set(class_id_to_name.values())


def get_class_names(ids):
    names = []
    for cls_id in ids:
        cls_name = class_id_to_name[str(cls_id)]
        names.append(cls_name)
    return np.asarray(names)


def get_train_instances():
    train_instances = ["w2_7f_corridor_01", "w2_9f_kitchen_10", "kurazume_lab", "morooka_office", "w2_2f_tatamiroom_02", "w2_10f_toilet_01"]

    return train_instances

