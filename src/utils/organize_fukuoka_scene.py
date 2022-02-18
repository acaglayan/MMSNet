import os
import pathlib
import shutil

from basic_utils import Constants, DataTypesFukuoka


def fukuoka_copy(png_depth_file_count, category, instance, instance_path, eval_root):
    indices = range(1, png_depth_file_count + 1)
    n_cat_name = category[:-1]
    if n_cat_name == "studyroom":
        n_cat_name = "study_room"

    print("Copying files under {}".format(instance_path))
    print("There are {} files copying...".format(len(indices)))

    for i in indices:
        fname_prefix = n_cat_name + Constants.DELIM + instance + Constants.DELIM
        rgb_file_name = "image{:04d}".format(i) + ".png"
        depth_file_name = "depth_image{:04d}".format(i) + ".png"
        txt_file_name = "depth_image{:04d}".format(i) + ".txt"

        n_rgb_file_name = DataTypesFukuoka.RGB + "{:04d}".format(i) + ".png"
        n_depth_file_name = DataTypesFukuoka.Depth + "{:04d}".format(i) + ".png"
        n_txt_file_name = DataTypesFukuoka.Depth + "{:04d}".format(i) + ".txt"

        shutil.copy(instance_path + rgb_file_name, eval_root + fname_prefix + n_rgb_file_name)
        shutil.copy(instance_path + depth_file_name, eval_root + fname_prefix + n_depth_file_name)
        shutil.copy(instance_path + txt_file_name, eval_root + fname_prefix + n_txt_file_name)


# We downloaded Fukuoka RGB-D Indoor Scene Dataset from the given url links below:
# Corridors: http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/corridors.tar.gz        267.5 MB
# Kitchens: http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/kitchens.tar.gz          214.1 MB
# Labs: http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/labs.tar.gz                  611.8 MB
# Offices: http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/offices.tar.gz            99.4 MB
# Study_rooms: http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/studyrooms.tar.gz     343.7 MB
# Toilets: http://robotics.ait.kyushu-u.ac.jp/~kurazume/data_research/toilets.tar.gz            121.4 MB

def organize_dataset():
    root_path = "???/fukuoka/public_set/"	# custom paths
    eval_root = "???/fukuoka/eval-set/"

    if not os.path.exists(eval_root):
        os.makedirs(eval_root)

    for category in sorted(os.listdir(root_path)):
        category_path = root_path + category + "/"
        if os.path.isdir(category_path):
            for instance in sorted(os.listdir(category_path)):
                instance_path = category_path + instance + "/"
                png_depth_file_count = len(list(pathlib.Path(instance_path).glob("depth*.png")))
                if png_depth_file_count == 0:
                    for sub_instance in sorted(os.listdir(instance_path)):
                        sub_instance_path = instance_path + sub_instance + "/"
                        png_depth_file_count = len(list(pathlib.Path(sub_instance_path).glob("depth*.png")))
                        if png_depth_file_count == 0:  # is there sub-sub instance? e.g. kurazume_lab_01
                            continue
                        else:
                            rec_instance_path = instance + Constants.DELIM + sub_instance[len(instance) + 1:]
                            fukuoka_copy(png_depth_file_count, category, rec_instance_path, sub_instance_path,
                                         eval_root)
                else:
                    fukuoka_copy(png_depth_file_count, category, instance, instance_path, eval_root)


if __name__ == '__main__':
    organize_dataset()
