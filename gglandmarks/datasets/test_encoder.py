import pytest
import numpy as np
from encoder import Encoder

def test_classes():
    classes = [1, 4, 5, 3, 2, 9]
    
    encoder = Encoder(classes)
    encoded_classes = np.array([1, 2, 3, 4, 5, 9])

    assert all([a == b for a, b in zip(encoded_classes, encoder.classes_)])

def test_encode():
    classes = [1, 4, 5, 3, 2, 9]
    data = [1, 1, 4, 5, 5, 4, 2, 3, 9, 9]    
    encoder = Encoder(classes)
    expected_values = [
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ]
    
    actual = encoder.encode(data)
    assert np.array_equal(expected_values, actual)

def test_decode():
    classes = [1, 4, 5, 3, 2, 9]
    data = [1, 1, 4, 5, 5, 4, 2, 3, 9, 9]
    encoder = Encoder(classes)
    onehot_values = [
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
    ]

    inverse_data = encoder.decode(onehot_values)

    assert(inverse_data.size > 0)
    assert(np.array_equal(inverse_data, data))
    
if '__main__' == __name__:    
    pytest.main([__file__])