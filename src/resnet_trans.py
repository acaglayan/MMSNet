import logging
import os
import pickle

import numpy as np
import torch

import basic_utils
import model_utils
import recursive_nn
from basic_utils import PrForm


def get_init_rnns():
    init_rnns = {
        'layer1': [],
        'layer2': [],
        'layer3': [],
        'layer4': [],
        'layer5': [],
        'layer6': [],
        'layer7': []
    }
    return init_rnns


def get_model_structure():
    rnn_layer_inp = {
        'layer1': (64, 28, 28),  # <- 64 x 56 x 56
        'layer2': (64, 14, 14),  # <- 256 x 56 x 56
        'layer3': (64, 14, 14),  # <- 512 x 28 x 28
        'layer4': (64, 28, 28),  # <- 1024 x 14 x 14
        'layer5': (64, 14, 14),  # <- 1024 x 14 x 14
        'layer6': (64, 14, 14),  # <- 2048 x 7 x 7
        'layer7': (64, 8, 8)  # <- 2048
    }
    return rnn_layer_inp


def model_reduction_plan():
    # num_split, chunk_size, rfs
    model_reduction = {
        'layer1': [(4, 64, 28, 'reduce_rfs')],
        'layer2': [(4, 64, 56, 'reduce_map'), (16, 64, 14, 'reduce_rfs')],
        'layer3': [(8, 64, 28, 'reduce_map'), (4, 64, 14, 'reduce_rfs')],
        'layer4': [(4, 256, 14, 'reduce_map')],
        'layer5': [(16, 64, 14, 'reduce_map')],
        'layer6': [(8, 256, 7, 'reduce_map')],
        'layer7': [(1, 1, 1, None)]
    }
    return model_reduction


def reduce_rfs(weights, layer_feats, num_reducing):
    # check the size availability
    assert np.mod(layer_feats.shape[2], np.sqrt(num_reducing)) < 1e-15
    weight_len = int(np.sqrt(num_reducing))

    result = np.multiply(layer_feats, weights)
    t_avg_pool = torch.nn.AvgPool2d(kernel_size=weight_len, stride=weight_len)
    result = t_avg_pool(basic_utils.numpy2tensor(result, device=torch.device("cpu"))) * num_reducing

    return basic_utils.tensor2numpy(result)


def randomized_pool(weights, layer_inp, num_split):
    assert layer_inp.ndim == 4
    num_maps = layer_inp.shape[1]
    assert np.mod(num_maps, num_split) < 1e-15
    chunk_size = int(num_maps / num_split)
    rfs = (layer_inp.shape[2], layer_inp.shape[3])

    out_layer = np.multiply(layer_inp, weights)
    out_layer = np.reshape(out_layer, (out_layer.shape[0], num_split, chunk_size,) + rfs)
    out_layer = np.sum(out_layer, axis=1)

    return out_layer


def reduce_map(weights, layer_feats, num_split):
    rnn_inp = randomized_pool(weights, layer_feats, num_split=num_split)

    return rnn_inp


def reduce_inp(weights, layer_feats, num_split, opt):
    if opt == 'reduce_rfs':
        rnn_inp = reduce_rfs(weights, layer_feats, num_split)
    else:
        rnn_inp = reduce_map(weights, layer_feats, num_split)

    return rnn_inp


def generate_reduction_randoms():
    all_weights = get_init_rnns()
    model_reduction = model_reduction_plan()
    for layer in model_reduction.keys():
        weight = None
        for ind in range(0, len(model_reduction[layer])):
            num_split, chunk_size, rfs, opt = model_reduction[layer][ind]
            if num_split != 1:
                weight = model_utils.init_random_weights(num_split, chunk_size, (rfs, rfs), opt)

            all_weights[layer].append(weight)
    return all_weights


class ResNetEncoder:
    def __init__(self, params):
        self.params = params
        self.reduction_weights = self.reduction_random_weights()
        self.rnn_weights = self.rnn_random_weights()

    def reduction_random_weights(self):
        if self.params.reuse_randoms:
            save_load_dir = self.params.features_root + 'random_weights/'
            reduc_weights_file = save_load_dir + "resnet101" + '_reduction_random_weights.pkl'
            if not os.path.exists(save_load_dir):
                os.makedirs(save_load_dir)

            try:
                with open(reduc_weights_file, 'rb') as f:
                    all_weights = pickle.load(f)
                    return all_weights
            except Exception:
                print('{}{}Failed to load the reduction weights file! They are going to be created for the first '
                      'time!{} '.format(PrForm.YELLOW, PrForm.BOLD, PrForm.END_FORMAT))
                logging.info('The reduction weights are going to be saved into {}'.format(reduc_weights_file))
                all_weights = generate_reduction_randoms()
                with open(reduc_weights_file, 'wb') as f:
                    pickle.dump(all_weights, f, pickle.HIGHEST_PROTOCOL)
                return all_weights
            finally:
                f.close()
        else:
            return generate_reduction_randoms()

    def generate_rnn_randoms(self):
        rnn_all_layer_weights = {}
        model_structure = get_model_structure()
        for layer in model_structure.keys():
            weights = recursive_nn.init_random_weights(self.params.num_rnn, model_structure[layer])
            rnn_all_layer_weights[layer] = weights

        return rnn_all_layer_weights

    def rnn_random_weights(self):
        if self.params.reuse_randoms:
            save_load_dir = self.params.features_root + 'random_weights/'
            rnn_weights_file = save_load_dir + "resnet101" + '_rnn_random_weights.pkl'
            if not os.path.exists(save_load_dir):
                os.makedirs(save_load_dir)
            try:
                with open(rnn_weights_file, 'rb') as f:
                    rnn_all_weights = pickle.load(f)
                    return rnn_all_weights
            except IOError:
                print('{}{}Failed to load the RNN weights file! They are going to be created for the first time!{}'.
                      format(PrForm.YELLOW, PrForm.BOLD, PrForm.END_FORMAT))
                logging.info('The RNN weights are going to be saved into {}'.format(rnn_weights_file))
                rnn_all_weights = self.generate_rnn_randoms()
                with open(rnn_weights_file, 'wb') as f:
                    pickle.dump(rnn_all_weights, f, pickle.HIGHEST_PROTOCOL)
                return rnn_all_weights
            finally:
                f.close()
        else:
            return self.generate_rnn_randoms()

    def process_layer1(self, curr_inputs):
        curr_layer = 'layer1'
        model_reduction = model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = reduce_inp(weights, curr_inputs, num_split=num_split, opt=opt)
        assert np.shape(pro_inp)[1:4] == get_model_structure()[curr_layer]

        return pro_inp

    def process_layer2(self, curr_inputs):
        curr_layer = 'layer2'
        model_reduction = model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = reduce_inp(weights, curr_inputs, num_split=num_split, opt=opt)

        num_split, _, _, opt = model_reduction[curr_layer][1]
        weights = self.reduction_weights[curr_layer][1]
        pro_inp = reduce_inp(weights, pro_inp, num_split=num_split, opt=opt)

        assert np.shape(pro_inp)[1:4] == get_model_structure()[curr_layer]

        return pro_inp

    def process_layer3(self, curr_inputs):
        curr_layer = 'layer3'
        model_reduction = model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = reduce_inp(weights, curr_inputs, num_split=num_split, opt=opt)

        num_split, _, _, opt = model_reduction[curr_layer][1]
        weights = self.reduction_weights[curr_layer][1]
        pro_inp = reduce_inp(weights, pro_inp, num_split=num_split, opt=opt)

        assert np.shape(pro_inp)[1:4] == get_model_structure()[curr_layer]

        return pro_inp

    def process_layer4(self, curr_inputs):
        curr_layer = 'layer4'
        model_reduction = model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = reduce_inp(weights, curr_inputs, num_split=num_split, opt=opt)

        pro_inp = model_utils.reshape_4d(pro_inp, shape=(64, 28, 28))
        assert np.shape(pro_inp)[1:4] == get_model_structure()[curr_layer]

        return pro_inp

    def process_layer5(self, curr_inputs):
        curr_layer = 'layer5'
        model_reduction = model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = reduce_inp(weights, curr_inputs, num_split=num_split, opt=opt)
        assert np.shape(pro_inp)[1:4] == get_model_structure()[curr_layer]

        return pro_inp

    def process_layer6(self, curr_inputs):
        curr_layer = 'layer6'
        model_reduction = model_reduction_plan()
        num_split, _, _, opt = model_reduction[curr_layer][0]
        weights = self.reduction_weights[curr_layer][0]
        pro_inp = reduce_inp(weights, curr_inputs, num_split=num_split, opt=opt)

        pro_inp = model_utils.reshape_4d(pro_inp, shape=(64, 14, 14))
        assert np.shape(pro_inp)[1:4] == get_model_structure()[curr_layer]

        return pro_inp

    def process_layer7(self, curr_inputs):
        curr_layer = 'layer7'
        pro_inp = np.concatenate((curr_inputs, curr_inputs), axis=1)
        pro_inp = model_utils.reshape_4d(pro_inp, shape=(64, 8, 8))
        assert np.shape(pro_inp)[1:4] == get_model_structure()[curr_layer]

        return pro_inp

    def preprocess_layer(self, cur_layer, curr_layer_inp):
        if cur_layer == 'layer1':
            processed_inp = self.process_layer1(curr_layer_inp)
        elif cur_layer == 'layer2':
            processed_inp = self.process_layer2(curr_layer_inp)
        elif cur_layer == 'layer3':
            processed_inp = self.process_layer3(curr_layer_inp)
        elif cur_layer == 'layer4':
            processed_inp = self.process_layer4(curr_layer_inp)
        elif cur_layer == 'layer5':
            processed_inp = self.process_layer5(curr_layer_inp)
        elif cur_layer == 'layer6':
            processed_inp = self.process_layer6(curr_layer_inp)
        else:
            processed_inp = self.process_layer7(curr_layer_inp)

        return processed_inp
