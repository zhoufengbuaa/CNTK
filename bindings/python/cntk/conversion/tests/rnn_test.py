import pytest
import numpy as np
import cntk as C

TEST_CONFIG = [
    # (num_layers, bidirectional, recurrent_op
    (1, True,  'lstm'),
    (1, False, 'lstm'),
    (2, False, 'lstm'),
    (3, True,  'lstm'),
    (4, True,  'rnnReLU'),
    (4, False, 'rnnTanh'),
]

@pytest.mark.parametrize("num_layers, bidirectional, recurrent_op", TEST_CONFIG)
def test_from_cudnn(num_layers, bidirectional, recurrent_op, device_id):
    if device_id == -1:
        pytest.skip('only runs on GPU')

    input_dim = 5
    hidden_dim = 3
    data = [[[1,2,3,4,5],[2,2,3,4,5],[3,2,3,4,5]], [[4,2,3,4,5],[5,2,3,4,5]]]
    input_var = C.sequence.input(shape=(input_dim,))
    W = C.parameter((-1,hidden_dim,), init = C.glorot_uniform())
    cudnn_rnn = C.optimized_rnnstack(input_var, W, hidden_dim, num_layers, bidirectional=bidirectional, recurrent_op=recurrent_op)
    cudnn_out = cudnn_rnn.eval({input_var:data})
    
    converted_rnn = C.conversion.rnn.from_cudnn(cudnn_rnn, hidden_dim, num_layers, bidirectional, recurrent_op=recurrent_op)
    converted_out = converted_rnn.eval({input_var:data})
    
    assert all(np.isclose(cudnn_out[i], converted_out[i]).all() for i in range(len(converted_out)))