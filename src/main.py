import logging

import numpy as np
import torch
from torch import nn
from torchvision import models

import recursive_nn
import sunrgbd
from backbone_extraction import ResNetExtractor
from basic_utils import get_params, init_save_dirs, get_timestamp, init_logger, DataTypesSUNRGBD, PrForm, tensor2numpy, \
    squeeze_dic_arrays, dictionary_to_tensor, numpy2tensor, convert_listvars_to_array
from fukuoka_loader import FukuokaRGBDScene, fukuoka_loader
from model_utils import get_num_classes, calc_modality_magnitudes, calc_scores
from multimodal_net import MultiModalNet
from nyuv2_loader import NYUV2RGBDScene
from resnet_trans import ResNetEncoder, get_model_structure, get_init_rnns
from sunrgbd_loader import sunrgbd_loader, SUNRGBDScene


def print_eval_acc(outs, labels, modality):
    outs = convert_listvars_to_array(outs)
    preds = np.argmax(outs, axis=1)
    avg_res, true_preds, test_size, _ = calc_scores(labels, preds)
    print("[Eval] Average {} Accuracy: {:.1f} True_Preds/Test_Size: {}/{}".format(modality, avg_res, true_preds, test_size))
    logging.info(
        "[Eval] Average {} Accuracy: {:.2f} True_Preds/Test_Size: {}/{}".format(modality, avg_res, true_preds, test_size))


def eval_mmsnet(params):
    split = "eval"
    if params.dataset == "sunrgbd":
        rgbd_test_set = SUNRGBDScene(params, split=split, loader=sunrgbd_loader)
    elif params.dataset == "nyuv2":
        rgbd_test_set = NYUV2RGBDScene(params, split=split)
    else:
        rgbd_test_set = FukuokaRGBDScene(params, split=split, loader=fukuoka_loader)

    # "cuda" if torch.cuda.is_available() else "cpu" instead of that force for cuda
    device = torch.device("cuda")
    logging.info('Using device "{}"'.format(device))

    sun_num_classes = len(sunrgbd.class_names)

    rgb_backbone_file = params.features_root + 'models/resnet101_' + DataTypesSUNRGBD.RGB + '_best_checkpoint.pth'
    depth_backbone_file = params.features_root + 'models/resnet101_' + DataTypesSUNRGBD.Depth + '_best_checkpoint.pth'

    rgb_model_ft = models.resnet101()
    num_ftrs = rgb_model_ft.fc.in_features
    rgb_model_ft.fc = nn.Linear(num_ftrs, sun_num_classes)

    depth_model_ft = models.resnet101()
    depth_model_ft.fc = nn.Linear(num_ftrs, sun_num_classes)

    mms_model_file = params.features_root + 'models/' + params.dataset + '_mms_best_checkpoint.pth'
    num_classes = get_num_classes(params)

    mms_net = MultiModalNet(num_classes)

    try:
        rgb_checkpoint = torch.load(rgb_backbone_file, map_location=device)
        depth_checkpoint = torch.load(depth_backbone_file, map_location=device)
        mms_checkpoint = torch.load(mms_model_file, map_location=device)

        rgb_model_ft.load_state_dict(rgb_checkpoint)
        depth_model_ft.load_state_dict(depth_checkpoint)
        mms_net.load_state_dict(mms_checkpoint, strict=False)
    except Exception as e:
        print('{}{}Failed to load the backbone SUN RGBD models: {}{}'.format(PrForm.BOLD, PrForm.RED, e,
                                                                             PrForm.END_FORMAT))
        return

    rgb_model_ft = rgb_model_ft.to(device)
    rgb_model_ft = rgb_model_ft.eval()

    depth_model_ft = depth_model_ft.to(device)
    depth_model_ft = depth_model_ft.eval()

    mms_net = mms_net.to(device)
    mms_net = mms_net.eval()

    rgbd_test_loader = torch.utils.data.DataLoader(rgbd_test_set, param.batch_size, shuffle=False)

    rnn_trans = ResNetEncoder(params)
    model_structure = get_model_structure()

    rgbd_test_outs, rgb_test_outs, depth_test_outs, test_labels = [], [], [], []
    for rgb_inputs, depth_inputs, labels, filenames in rgbd_test_loader:
        rgb_inputs = rgb_inputs.to(device)
        depth_inputs = depth_inputs.to(device)

        rgb_rnn_encoded = get_init_rnns()
        depth_rnn_encoded = get_init_rnns()

        for extracted_layer in range(1, 8):
            curr_layer = 'layer' + str(extracted_layer)
            rgb_backbone = ResNetExtractor(rgb_model_ft, extracted_layer)
            depth_backbone = ResNetExtractor(depth_model_ft, extracted_layer)

            rgb_backbone_features = tensor2numpy(rgb_backbone(rgb_inputs))
            depth_backbone_features = tensor2numpy(depth_backbone(depth_inputs))

            rgb_backbone_features = rnn_trans.preprocess_layer(curr_layer, rgb_backbone_features)
            depth_backbone_features = rnn_trans.preprocess_layer(curr_layer, depth_backbone_features)

            rgb_rnn_out = recursive_nn.forward_rnn(rnn_trans.rnn_weights[curr_layer], rgb_backbone_features,
                                                   params.num_rnn, model_structure[curr_layer])
            depth_rnn_out = recursive_nn.forward_rnn(rnn_trans.rnn_weights[curr_layer], depth_backbone_features,
                                                     params.num_rnn, model_structure[curr_layer])

            rgb_rnn_encoded[curr_layer].append(rgb_rnn_out)
            depth_rnn_encoded[curr_layer].append(depth_rnn_out)

        rgb_rnn_encoded = squeeze_dic_arrays(rgb_rnn_encoded)
        depth_rnn_encoded = squeeze_dic_arrays(depth_rnn_encoded)

        converted_rgb_inputs = dictionary_to_tensor(rgb_rnn_encoded)
        converted_rgb_inputs = numpy2tensor(converted_rgb_inputs)

        converted_depth_inputs = dictionary_to_tensor(depth_rnn_encoded)
        converted_depth_inputs = numpy2tensor(converted_depth_inputs)

        rgb_outs, _, depth_outs, _ = mms_net(converted_rgb_inputs, converted_depth_inputs)

        w_rgb, w_depth = calc_modality_magnitudes((rgb_outs, depth_outs))
        rgbd_outs = w_rgb.unsqueeze(1) * rgb_outs + w_depth.unsqueeze(1) * depth_outs

        rgbd_test_outs.append(tensor2numpy(rgbd_outs))
        rgb_test_outs.append(tensor2numpy(rgb_outs))
        depth_test_outs.append(tensor2numpy(depth_outs))
        test_labels.append(labels)

    test_labels = convert_listvars_to_array(test_labels)
    print_eval_acc(rgbd_test_outs, test_labels, modality="RGBD")
    print_eval_acc(rgb_test_outs, test_labels, modality="RGB")
    print_eval_acc(depth_test_outs, test_labels, modality="Depth")


if __name__ == '__main__':
    param = get_params()
    param = init_save_dirs(param)
    logfile_name = param.log_dir + get_timestamp() + '_MMSNet_eval.log'
    init_logger(logfile_name, param)

    eval_mmsnet(param)
