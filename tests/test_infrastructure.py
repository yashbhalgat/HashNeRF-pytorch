"""Test to validate the testing infrastructure setup."""

import pytest
import torch
import numpy as np
from pathlib import Path


def test_pytest_working():
    """Test that pytest is working correctly."""
    assert True


def test_torch_available():
    """Test that PyTorch is available and working."""
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum() == 6


def test_numpy_available():
    """Test that NumPy is available and working."""
    array = np.array([1, 2, 3])
    assert array.sum() == 6


def test_cuda_detection():
    """Test CUDA detection (should work whether CUDA is available or not)."""
    cuda_available = torch.cuda.is_available()
    assert isinstance(cuda_available, bool)


def test_fixtures_working(temp_dir, device, sample_config):
    """Test that shared fixtures are working."""
    # Test temp directory fixture
    assert temp_dir.exists()
    assert temp_dir.is_dir()
    
    # Test device fixture
    assert isinstance(device, torch.device)
    
    # Test config fixture
    assert isinstance(sample_config, dict)
    assert 'datadir' in sample_config
    assert 'dataset_type' in sample_config


def test_bounding_box_fixture(sample_bounding_box):
    """Test bounding box fixture."""
    assert sample_bounding_box.shape == (2, 3)
    assert torch.is_tensor(sample_bounding_box)


def test_rays_fixture(sample_rays):
    """Test rays fixture."""
    rays_o, rays_d = sample_rays
    assert rays_o.shape[1] == 3  # Origin vectors
    assert rays_d.shape[1] == 3  # Direction vectors
    assert rays_o.shape[0] == rays_d.shape[0]  # Same batch size
    
    # Check that direction vectors are normalized
    norms = torch.norm(rays_d, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_points_fixture(sample_points):
    """Test points fixture."""
    assert sample_points.shape[1] == 3
    assert torch.is_tensor(sample_points)


def test_image_data_fixture(sample_image_data):
    """Test image data fixture."""
    images, poses, hwf = sample_image_data
    assert images.shape[-1] == 3  # RGB channels
    assert poses.shape[-2:] == (4, 4)  # 4x4 transformation matrices
    assert len(hwf) == 3  # Height, width, focal length


def test_mock_nerf_model(mock_nerf_model):
    """Test mock NeRF model fixture."""
    # Test forward pass
    input_tensor = torch.randn(10, 63)
    output = mock_nerf_model(input_tensor)
    assert output.shape == (10, 4)


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker works."""
    assert True


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works."""
    import time
    time.sleep(0.1)  # Small delay to simulate slow test
    assert True


def test_project_structure():
    """Test that project structure is correct."""
    project_root = Path(__file__).parent.parent
    
    # Check main files exist
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "run_nerf.py").exists()
    assert (project_root / "hash_encoding.py").exists()
    
    # Check test structure
    tests_dir = project_root / "tests"
    assert tests_dir.exists()
    assert (tests_dir / "__init__.py").exists()
    assert (tests_dir / "conftest.py").exists()
    assert (tests_dir / "unit" / "__init__.py").exists()
    assert (tests_dir / "integration" / "__init__.py").exists()


class TestInfrastructureClass:
    """Test class to verify class-based testing works."""
    
    def test_class_method(self):
        """Test method within a test class."""
        assert True
    
    def test_class_with_fixture(self, device):
        """Test class method with fixture."""
        assert isinstance(device, torch.device)