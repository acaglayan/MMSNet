import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchvision import transforms

import depth_transform
import fukuoka
import nyuv2
import sunrgbd
from basic_utils import DataTypesSUNRGBD, DataTypesFukuoka, DataTypesNYUV2


def init_random_weights(num_split, chunk_size, rfs, opt):
    if opt == 'reduce_rfs':
        num_map = chunk_size
        rfs = int(np.sqrt(num_split)) * rfs[0]
        rfs = (rfs, rfs)
    else:
        num_map = chunk_size * num_split

    weights = np.zeros(shape=(num_map,) + rfs, dtype=np.float32)
    for i in range(num_map):
        random_weight = -0.1 + 0.2 * np.random.rand(rfs[0], rfs[1])
        weights[i, :] = random_weight

    return weights


def get_data_transform(data_type):
    std = [0.23089603, 0.2393163, 0.23400005]
    if data_type in (DataTypesSUNRGBD.RGB, DataTypesFukuoka.RGB, DataTypesNYUV2.RGB):
        mean = [0.15509185, 0.16330947, 0.1496393]
        data_form = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:  # type(data_type) is np.ndarray
        mean = [0.0, 0.0, 0.0]  # [0.485, 0.456, 0.406]
        data_form = depth_transform.Compose([
            depth_transform.Resize(size=(256, 256), interpolation='NEAREST'),
            depth_transform.CenterCrop(224),
            depth_transform.ToTensor(),
            depth_transform.Normalize(mean, std)
        ])

    return data_form


def reshape_4d(layer_feats, shape):
    layer_feats = np.reshape(layer_feats, (layer_feats.shape[0],) + shape)

    return layer_feats


def get_num_classes(params):
    if params.dataset == "sunrgbd":
        num_classes = len(sunrgbd.class_names)
    elif params.dataset == "nyuv2":
        num_classes = len(nyuv2.class_names)
    else:
        num_classes = len(fukuoka.class_names)

    return num_classes


def calc_modality_magnitudes(modality_outs):
    assert len(modality_outs) == 2
    l1_scores = modality_outs[0]
    l2_scores = modality_outs[1]

    s_l1 = torch.sum(torch.square(l1_scores), dim=1)
    s_l2 = torch.sum(torch.square(l2_scores), dim=1)

    m_l1 = s_l1 / torch.maximum(s_l1, s_l2)
    m_l2 = s_l2 / torch.maximum(s_l1, s_l2)

    w_l1 = torch.sqrt(torch.exp(m_l1) / (torch.exp(m_l1) + torch.exp(m_l2)))
    w_l2 = 1 - w_l1

    return w_l1, w_l2


def calc_scores(test_labels, preds):
    result = (preds == test_labels)
    avg_res = np.mean(result) * 100
    true_preds = np.count_nonzero(result == True)
    test_size = np.size(result)
    conf_mat = confusion_matrix(test_labels, preds)
    return avg_res, true_preds, test_size, conf_mat
