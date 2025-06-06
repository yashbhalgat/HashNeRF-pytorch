"""Sample integration test to validate integration testing structure."""

import pytest
import torch


@pytest.mark.integration
def test_torch_cuda_integration():
    """Test PyTorch CUDA integration (if available)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        result = tensor * 2
        expected = torch.tensor([2.0, 4.0, 6.0]).to(device)
        assert torch.allclose(result, expected)
    else:
        pytest.skip("CUDA not available")


@pytest.mark.integration
def test_model_device_integration(device, mock_nerf_model):
    """Test model and device integration."""
    model = mock_nerf_model.to(device)
    input_tensor = torch.randn(5, 63).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    assert output.device == device
    assert output.shape == (5, 4)