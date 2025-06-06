"""Shared pytest fixtures for HashNeRF testing."""

import os
import tempfile
import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'datadir': './data/nerf_synthetic/lego',
        'dataset_type': 'blender',
        'white_bkgd': True,
        'half_res': True,
        'testskip': 8,
        'N_samples': 64,
        'N_importance': 64,
        'use_viewdirs': True,
        'raw_noise_std': 0.0,
        'lrate': 5e-4,
        'lrate_decay': 250,
        'chunk': 1024*32,
        'netchunk': 1024*64,
        'no_batching': True,
        'no_reload': False,
        'ft_path': None,
        'N_iters': 200000,
        'i_print': 100,
        'i_img': 500,
        'i_weights': 10000,
        'i_testset': 50000,
        'i_video': 50000,
        'N_rand': 1024,
        'precrop_iters': 0,
        'precrop_frac': 0.5,
        'multires': 10,
        'multires_views': 4,
        'netdepth': 8,
        'netwidth': 256,
        'netdepth_fine': 8,
        'netwidth_fine': 256,
        'perturb': 1.0,
        'lindisp': False,
        'i_embed': 0,
        'render_only': False,
        'render_test': False,
        'render_factor': 0,
        'expname': 'test',
        'basedir': './logs',
        'factor': 8,
        'no_ndc': True,
        'spherify': False,
        'llffhold': 8,
        'bound': 1.5,
        'finest_res': 512,
        'log2_hashmap_size': 19,
        'lrate_hash': 1e-2,
        'hash_levels': 16,
        'hash_features': 2
    }


@pytest.fixture
def sample_bounding_box():
    """Sample bounding box for testing."""
    return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)


@pytest.fixture
def sample_rays():
    """Sample ray data for testing."""
    batch_size = 1024
    rays_o = torch.randn(batch_size, 3)
    rays_d = torch.randn(batch_size, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # normalize
    return rays_o, rays_d


@pytest.fixture
def sample_points():
    """Sample 3D points for testing."""
    return torch.randn(1000, 3)


@pytest.fixture
def sample_image_data():
    """Sample image data for testing."""
    height, width = 100, 100
    images = np.random.rand(10, height, width, 3).astype(np.float32)
    poses = np.random.rand(10, 4, 4).astype(np.float32)
    hwf = [height, width, 100.0]  # focal length
    return images, poses, hwf


@pytest.fixture
def mock_config_file(temp_dir):
    """Create a mock config file for testing."""
    config_content = """expname = test_experiment
basedir = ./logs
datadir = ./data/nerf_synthetic/lego
dataset_type = blender
white_bkgd = True
half_res = True
testskip = 8
"""
    config_file = temp_dir / "test_config.txt"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def mock_nerf_model():
    """Mock NeRF model for testing."""
    import torch.nn as nn
    
    class MockNeRF(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(63, 4)  # 63 = 3*21 for position encoding
            
        def forward(self, x):
            return self.linear(x)
    
    return MockNeRF()


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files."""
    return Path(__file__).parent / "test_data"