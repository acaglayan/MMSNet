import numpy as np

class_id_to_name = {
    "0": "bathroom",
    "1": "bedroom",
    "2": "bookstore",
    "3": "classroom",
    "4": "dining_room",
    "5": "home_office",
    "6": "kitchen",
    "7": "living_room",
    "8": "office",
    "9": "others"
}
class_name_to_id = {v: k for k, v in class_id_to_name.items()}

class_names = set(class_id_to_name.values())


def get_class_names(ids):
    names = []
    for cls_id in ids:
        cls_name = class_id_to_name[str(cls_id)]
        names.append(cls_name)
    return np.asarray(names)

