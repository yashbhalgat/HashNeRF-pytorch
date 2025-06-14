import pytest
import tempfile
import shutil
import os
import json
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file path."""
    def _temp_file(filename="test_file.txt"):
        return os.path.join(temp_dir, filename)
    return _temp_file


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        "expname": "test_experiment",
        "basedir": "./logs",
        "datadir": "./data/test",
        "N_rand": 1024,
        "N_samples": 64,
        "N_importance": 128,
        "perturb": 1.0,
        "use_viewdirs": True,
        "i_embed": 0,
        "multires": 10,
        "multires_views": 4,
        "raw_noise_std": 0.0,
        "render_only": False,
        "render_test": False,
        "render_factor": 0,
        "precrop_iters": 0,
        "precrop_frac": 0.5,
        "dataset_type": "blender",
        "testskip": 8,
        "shape": "greek",
        "white_bkgd": False,
        "half_res": False,
        "factor": 8,
        "no_ndc": True,
        "lindisp": False,
        "spherify": False,
        "llffhold": 8,
        "i_print": 100,
        "i_img": 500,
        "i_weights": 10000,
        "i_testset": 50000,
        "i_video": 50000,
        "N_iters": 200000,
        "finest_res": 512,
        "log2_hashmap_size": 19,
        "sparse_loss_weight": 0.0,
        "tv_loss_weight": 0.0,
        "lrate": 0.01,
        "lrate_decay": 10,
        "chunk": 1024*32,
        "netchunk": 1024*64,
        "no_batching": False,
        "no_reload": False,
        "ft_path": None,
        "random_seed": None
    }


@pytest.fixture
def sample_images():
    """Create sample image tensors for testing."""
    batch_size = 4
    height, width = 100, 100
    channels = 3
    images = torch.rand(batch_size, height, width, channels)
    return images


@pytest.fixture
def sample_poses():
    """Create sample camera pose matrices."""
    num_poses = 10
    poses = torch.eye(4).unsqueeze(0).repeat(num_poses, 1, 1)
    poses[:, :3, 3] = torch.randn(num_poses, 3)
    return poses


@pytest.fixture
def sample_rays():
    """Create sample ray origins and directions."""
    num_rays = 1000
    rays_o = torch.randn(num_rays, 3)
    rays_d = torch.randn(num_rays, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    return rays_o, rays_d


@pytest.fixture
def mock_model():
    """Create a mock neural network model."""
    model = Mock()
    model.forward = MagicMock(return_value=torch.randn(100, 4))
    model.parameters = MagicMock(return_value=[torch.randn(10, 10)])
    return model


@pytest.fixture
def sample_training_data():
    """Create sample training data for NeRF."""
    return {
        "images": torch.rand(100, 100, 100, 3),
        "poses": torch.eye(4).unsqueeze(0).repeat(100, 1, 1),
        "render_poses": torch.eye(4).unsqueeze(0).repeat(40, 1, 1),
        "hwf": [100, 100, 50.0],
        "i_split": [[0, 80], [80, 90], [90, 100]]
    }


@pytest.fixture
def mock_hash_encoding():
    """Create a mock hash encoding module."""
    mock = Mock()
    mock.n_levels = 16
    mock.n_features_per_level = 2
    mock.forward = MagicMock(return_value=torch.randn(1000, 32))
    return mock


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture
def sample_config_file(temp_dir):
    """Create a sample configuration file."""
    config_path = os.path.join(temp_dir, "test_config.txt")
    config_content = """
expname = test_experiment
basedir = ./logs
datadir = ./data/nerf_synthetic/chair

dataset_type = blender
no_batching = True

use_viewdirs = True
finest_res = 512
log2_hashmap_size = 19

N_samples = 64
N_importance = 64

perturb = 1.
raw_noise_std = 0.

render_only = False
render_test = False

chunk = 32768
netchunk = 65536

lrate = 0.01
lrate_decay = 10

N_iters = 30000
i_testset = 2500
i_video = 10000
i_print = 100
"""
    with open(config_path, 'w') as f:
        f.write(config_content)
    return config_path


@pytest.fixture
def mock_dataloader():
    """Create a mock data loader."""
    loader = Mock()
    loader.__iter__ = MagicMock(return_value=iter([
        (torch.randn(32, 3), torch.randn(32, 3), torch.randn(32))
        for _ in range(10)
    ]))
    loader.__len__ = MagicMock(return_value=10)
    return loader


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """Clean up GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture log messages during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog