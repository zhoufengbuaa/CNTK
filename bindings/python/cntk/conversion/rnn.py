# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import numpy as np
import cntk as C

def from_cudnn(cudnn_rnn, hidden_size, num_layers, bidirectional, recurrent_op):
    '''
    from_cudnn(cudnn_rnn, hidden_size, num_layers)

    converts cudnn optimized_rnnstack to non-cudnn functions to run in non-CUDA environment
    '''
    if recurrent_op != 'lstm':
        raise(ValueError('unsupported recurrent_op value "%s"'%recurrent_op))
    #note that cudnn GRU is different from standard GRU so no conversion unless creating a new type of GRU cell for CPU

    if cudnn_rnn.root_function.op_name != 'OptimizedRNNStack':
        raise(ValueError('unexpected cudnn_rnn.root_function.op_name value "%s"'%cudnn_rnn.root_function.op_name))
    
    cudnn_param = cudnn_rnn.parameters[0]
    input_var = cudnn_rnn.arguments[0]
    
    def _any_inferred(shape):
        return np.any([dim < 0 for dim in shape])
    
    if _any_inferred(cudnn_param.shape) or _any_inferred(input_var.shape):
        raise(ValueError('parameter not initialized yet'))

    input_size = input_var.shape[0] if len(input_var.shape) else 1
    
    num_gates = 1
    rnn_lambda = None
    if recurrent_op == 'lstm':
        num_gates = 4
        if bidirectional:
            rnn_lambda = lambda x, i : C.splice(C.layers.Recurrence(C.layers.LSTM(hidden_size, name='rnn_fw'+i))(x), C.layers.Recurrence(C.layers.LSTM(hidden_size, name='rnn_bw'+i), go_backwards=True)(x))
        else:
            rnn_lambda = lambda x, i : C.layers.Recurrence(C.layers.LSTM(hidden_size, name="rnn_"+i))(x)

    noncudnn_func = rnn_lambda(input_var, '0')

    param = cudnn_param.value.reshape(-1)
    offset = 0
    multiplier = 2 if bidirectional else 1

    def _adjust_gate_order(W):
        if recurrent_op == 'lstm':
            if len(W.shape) == 2:
                i,f,m,o = np.hsplit(W, 4)
                return np.concatenate((i,m,f,o), axis=1)
            elif len(W.shape) == 1:
                i,f,m,o = np.split(W, 4)
                return np.concatenate((i,m,f,o))
            else:
                raise Exception('invalid input')
        else:
            return W

    def _get_cudnn_rnn_weight_splitter(in_dim, h_dim):
        # for unidirectional, W, H
        # for bidirectional, fw_W, fw_H, bw_W, bw_H
        splitter = [in_dim*h_dim*num_gates, h_dim*h_dim*num_gates] * multiplier
        splitter = splitter[0:-1]
        return np.cumsum(splitter)

    def _get_cudnn_rnn_bias_splitter(h_dim):
        # for unidirectional, b1, b2
        # for bidirectional, fw_b1, fw_b2, bw_b1, bw_b2
        splitter = [h_dim*num_gates, h_dim*num_gates] * multiplier
        splitter = splitter[0:-1]
        return np.cumsum(splitter)

    offset = 0
    layer_input_size = input_size
    for layer in range(num_layers):
        layer_size = (layer_input_size + hidden_size) * hidden_size * num_gates * multiplier
        layer_param = param[offset:offset+layer_size]
        layer_name = '{}'.format(layer)
        if bidirectional:
            fw_Wt, fw_Ht, bw_Wt, bw_Ht = np.split(layer_param, _get_cudnn_rnn_weight_splitter(layer_input_size, hidden_size))
            fw_cell = noncudnn_func.find_by_name('rnn_fw'+layer_name, -1)
            bw_cell = noncudnn_func.find_by_name('rnn_bw'+layer_name, -1)
            fw_cell.W.value = _adjust_gate_order(fw_Wt.reshape(num_gates*hidden_size, -1).transpose())
            fw_cell.H.value = _adjust_gate_order(fw_Ht.reshape(num_gates*hidden_size, -1).transpose())
            bw_cell.W.value = _adjust_gate_order(bw_Wt.reshape(num_gates*hidden_size, -1).transpose())
            bw_cell.H.value = _adjust_gate_order(bw_Ht.reshape(num_gates*hidden_size, -1).transpose())
        else:
            Wt, Ht = np.split(layer_param, _get_cudnn_rnn_weight_splitter(layer_input_size, hidden_size))
            cell = noncudnn_func.find_by_name('rnn_'+layer_name, -1)
            cell.W.value = _adjust_gate_order(Wt.reshape(num_gates*hidden_size, -1).transpose())
            cell.H.value = _adjust_gate_order(Ht.reshape(num_gates*hidden_size, -1).transpose())

        offset += layer_size
        layer_input_size = hidden_size * multiplier
        
        if layer != num_layers - 1:
            noncudnn_func = rnn_lambda(noncudnn_func.output, '{}'.format(layer+1))

    for layer in range(num_layers):
        layer_size = 2 * hidden_size * num_gates * multiplier
        layer_param = param[offset:offset+layer_size]
        layer_name = '{}'.format(layer)
        if bidirectional:
            fw_b1, fw_b2, bw_b1, bw_b2 = np.split(layer_param, _get_cudnn_rnn_bias_splitter(hidden_size))
            fw_cell = noncudnn_func.find_by_name('rnn_fw'+layer_name, -1)
            bw_cell = noncudnn_func.find_by_name('rnn_bw'+layer_name, -1)
            fw_cell.b.value = _adjust_gate_order(fw_b1 + fw_b2).reshape(-1)
            bw_cell.b.value = _adjust_gate_order(bw_b1 + bw_b2).reshape(-1)
        else:
            b1, b2 = np.split(layer_param, _get_cudnn_rnn_bias_splitter(hidden_size))
            cell = noncudnn_func.find_by_name('rnn_'+layer_name, -1)
            cell.b.value = _adjust_gate_order(b1 + b2).reshape(-1)
        offset += layer_size

    return noncudnn_func