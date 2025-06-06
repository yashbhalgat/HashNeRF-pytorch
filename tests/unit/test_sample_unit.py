"""Sample unit test to validate unit testing structure."""

import pytest
import torch
import numpy as np


@pytest.mark.unit
def test_tensor_operations():
    """Test basic tensor operations."""
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    result = a + b
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(result, expected)


@pytest.mark.unit
def test_numpy_operations():
    """Test basic numpy operations."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    result = np.dot(a, b)
    expected = 32  # 1*4 + 2*5 + 3*6
    assert result == expected


@pytest.mark.unit
def test_with_mock(mocker):
    """Test using pytest-mock functionality."""
    mock_func = mocker.Mock(return_value=42)
    result = mock_func("test_arg")
    assert result == 42
    mock_func.assert_called_once_with("test_arg")