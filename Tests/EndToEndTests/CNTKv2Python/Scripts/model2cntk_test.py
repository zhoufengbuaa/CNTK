# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import pytest
import cntk
import numpy as np
import os
import sys

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", ))

from unimodel import cntkmodel
from unimodel import cntkinstance
from utils import globalconf


def tensor_initializer(tensor_size):
    tensor = cntkmodel.CntkTensorDefinition()
    tensor.tensor = tensor_size
    tensor.data = np.ones(tensor_size, np.float32)
    return tensor

STANDARD_DATA = [
    (3, 8, 8)
]

CONVOLUTION_DATA = [
    ((3, 8, 8),
     (3, 3),
     (2, 2),
     (1, 1),
     16, 1, True, False,
     (16, 4, 4))
]


@pytest.mark.parametrize('input_size, kernel_size, stride, dilation, out_channel, group, '
                         'pad, bias, output_size', CONVOLUTION_DATA)
def test_convolution_ops(input_size, kernel_size, stride, dilation, out_channel, group, pad, bias, output_size):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False)]
    # define conv parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    conv_param = cntkmodel.CntkConvolutionParameters()
    conv_param.dilation = dilation
    conv_param.group = group
    conv_param.auto_pad = pad
    conv_param.kernel = kernel_size
    conv_param.stride = stride
    conv_param.need_bias = bias
    conv_param.output = out_channel
    cntk_layer_def.parameters = conv_param
    cntk_layer_def.op_name = 'conv_test'
    cntk_layer_def.parameter_tensor = [tensor_initializer((out_channel, input_size[0]) + kernel_size)]
    if bias:
        cntk_layer_def.parameter_tensor.append(tensor_initializer(out_channel, ))
    instance = cntkinstance.ApiSetup.convolution(cntk_layer_def, ops_input)
    assert (instance.output.shape == output_size)

POOLING_DATA = [
    ((3, 8, 8),
     (3, 3),
     (2, 2),
     False,
     (3, 4, 4))
]


@pytest.mark.parametrize('input_size, kernel_size, stride, pad, output_size', POOLING_DATA)
def test_pooling_ops(input_size, kernel_size, stride, pad, output_size):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False)]
    # define pooling parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    pooling_param = cntkmodel.CntkPoolingParameters()
    pooling_param.auto_pad = pad
    pooling_param.kernel = kernel_size
    pooling_param.stride = stride
    pooling_param.pooling_type = 0
    cntk_layer_def.parameters = pooling_param
    cntk_layer_def.op_name = 'pooling_test'
    instance = cntkinstance.ApiSetup.pooling(cntk_layer_def, ops_input)
    assert (instance.output.shape == output_size)


@pytest.mark.parametrize('input_size', STANDARD_DATA)
def test_batchnorm_ops(input_size):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False)]
    # define batch norm parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    cntk_layer_def.parameters = cntkmodel.CntkBatchNormParameters()
    cntk_layer_def.parameter_tensor = [tensor_initializer((input_size[0], ))] * 4
    cntk_layer_def.op_name = 'batch_norm_test'
    instance = cntkinstance.ApiSetup.batch_normalization(cntk_layer_def, ops_input)
    assert (instance.output.shape == input_size)

LRN_DATA = [
    ((16, 5, 5),
     3)
]


@pytest.mark.parametrize('input_size, kernel_size', LRN_DATA)
def test_lrn_ops(input_size, kernel_size):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False)]
    # define lrn parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    cntk_layer_def.parameters = cntkmodel.CntkLRNParameters()
    cntk_layer_def.parameters.kernel_size = kernel_size
    cntk_layer_def.op_name = 'lrn_test'
    instance = cntkinstance.ApiSetup.lrn(cntk_layer_def, ops_input)
    assert (instance.output.shape == input_size)


@pytest.mark.parametrize('input_size', STANDARD_DATA)
def test_relu_ops(input_size):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False)]
    # define relu parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    cntk_layer_def.parameters = cntkmodel.CntkParameters()
    cntk_layer_def.op_name = 'relu_test'
    instance = cntkinstance.ApiSetup.relu(cntk_layer_def, ops_input)
    assert (instance.output.shape == input_size)

LINEAR_DATA = [
    ((10, 1, 1),
     32)
]


@pytest.mark.parametrize('input_size, out_channel', LINEAR_DATA)
def test_linear_ops(input_size, out_channel):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False)]
    # define linear parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    linear_params = cntkmodel.CntkLinearLayerParameters()
    linear_params.num_output = out_channel
    cntk_layer_def.parameters = linear_params
    cntk_layer_def.parameter_tensor = [tensor_initializer((out_channel, np.prod(input_size)))]
    cntk_layer_def.parameter_tensor += [tensor_initializer((out_channel,))]
    cntk_layer_def.op_name = 'linear_test'
    instance = cntkinstance.ApiSetup.linear(cntk_layer_def, ops_input)
    assert (np.prod(instance.output.shape) == out_channel)


@pytest.mark.parametrize('input_size', STANDARD_DATA)
def test_plus_ops(input_size):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False),
                 cntk.input(input_size, needs_gradient=False)]
    # define plus parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    cntk_layer_def.parameters = cntkmodel.CntkParameters()
    cntk_layer_def.op_name = 'plus_test'
    instance = cntkinstance.ApiSetup.plus(cntk_layer_def, ops_input)
    assert (instance.output.shape == input_size)

SPLICE_DATA = [
    ([(3, 8, 8), (7, 8, 8)],
     (10, 8, 8))
]


@pytest.mark.parametrize('input_size, output_size', SPLICE_DATA)
def test_splice_ops(input_size, output_size):
    # define input variable
    ops_input = [cntk.input(in_size, needs_gradient=False) for in_size in input_size]
    # define splice parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    cntk_layer_def.parameters = cntkmodel.CntkParameters()
    cntk_layer_def.op_name = 'splice_test'
    instance = cntkinstance.ApiSetup.splice(cntk_layer_def, ops_input)
    assert (instance.output.shape == output_size)

SOFTMAX_DATA = [
    (1000, )
]


@pytest.mark.parametrize('input_size', SOFTMAX_DATA)
def test_softmax_ops(input_size):
    # define input variable
    ops_input = [cntk.input(input_size, needs_gradient=False)]
    # define softmax parameters
    cntk_layer_def = cntkmodel.CntkLayersDefinition()
    cntk_layer_def.parameters = cntkmodel.CntkParameters()
    cntk_layer_def.op_name = 'softmax_test'
    instance = cntkinstance.ApiSetup.softmax(cntk_layer_def, ops_input)
    assert (instance.output.shape == input_size)


def test_eval_converted_model():
    model = cntk.load_model('NIN_cifar.cntkmodel')
    cfg = globalconf.load_conf('global.json')
    evaluate = cntkinstance.Evaluator(global_conf=cfg, models=model)
    evaluate.eval_model()
