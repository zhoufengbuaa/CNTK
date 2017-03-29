# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
from cntk import *
from cntk.layers import *
from cntk.layers.typing import *

import pytest

@pytest.mark.parametrize("layers_count, dense_units", [(4,5), (6,9), (7, 10)])
def test_for_constructor_layer(layers_count, dense_units):
    x = Input(4)

    network = For(range(layers_count), lambda i: Dense(dense_units))

    expected_num_of_parameters = 2 * layers_count
    assert len(network.parameters) == expected_num_of_parameters

    res = network(x)

    expected_output_shape = (dense_units,)
    assert res.shape == expected_output_shape

@pytest.mark.parametrize("input_data", [[2, 8],[4, 7, 9], [5, 6, 10]])
def test_sequential_clique(input_data):
    x = Input(len(input_data))

    seq_clique = SequentialClique([abs, sqrt, square])(x)

    assert seq_clique.shape == x.shape

    np_data = np.asarray(input_data, np.float32)
    res = seq_clique.eval(np_data)

    expected_res = np.abs(np_data) + np_data
    expected_res += np.sqrt(expected_res)
    expected_res = np.square(expected_res)

    expected_res.shape = (1, 1) + expected_res.shape

    np.testing.assert_array_almost_equal(res, expected_res, decimal=4)

@pytest.mark.parametrize("input_data", [[3, 5],[9, 25, 13]])
def test_resnet_block(input_data):
    x = Input(len(input_data))

    res_net = ResNetBlock(square)(x)

    np_data = np.asarray(input_data, np.float32)

    actual_res = res_net.eval(np_data)

    expected_res = np.square(np_data) + np_data
    expected_res.shape = (1, 1) + expected_res.shape

    np.testing.assert_array_equal(actual_res, expected_res)

