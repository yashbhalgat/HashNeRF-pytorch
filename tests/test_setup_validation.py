import pytest
import sys
import os
from pathlib import Path


class TestSetupValidation:
    """Validation tests to ensure the testing infrastructure is properly configured."""
    
    def test_python_version(self):
        """Test that Python version meets requirements."""
        assert sys.version_info >= (3, 8), "Python 3.8 or higher is required"
    
    def test_project_structure(self):
        """Test that the project structure is set up correctly."""
        project_root = Path(__file__).parent.parent
        
        # Check main directories exist
        assert project_root.exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        
        # Check __init__.py files
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit" / "__init__.py").exists()
        assert (project_root / "tests" / "integration" / "__init__.py").exists()
        
        # Check configuration files
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
    
    def test_conftest_fixtures(self):
        """Test that conftest fixtures are available."""
        # These imports should work if conftest.py is properly set up
        from tests.conftest import (
            temp_dir, temp_file, mock_config, sample_images,
            sample_poses, sample_rays, mock_model
        )
        assert True  # If we get here, imports worked
    
    def test_imports(self):
        """Test that main project modules can be imported."""
        try:
            # Test importing main modules
            import run_nerf
            import hash_encoding
            import run_nerf_helpers
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that the unit test marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that the integration test marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that the slow test marker works."""
        assert True
    
    def test_fixture_temp_dir(self, temp_dir):
        """Test the temp_dir fixture."""
        assert os.path.exists(temp_dir)
        assert os.path.isdir(temp_dir)
        
        # Create a test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        assert os.path.exists(test_file)
    
    def test_fixture_mock_config(self, mock_config):
        """Test the mock_config fixture."""
        assert isinstance(mock_config, dict)
        assert "expname" in mock_config
        assert "basedir" in mock_config
        assert "datadir" in mock_config
        assert mock_config["expname"] == "test_experiment"
    
    def test_fixture_sample_images(self, sample_images):
        """Test the sample_images fixture."""
        import torch
        
        assert isinstance(sample_images, torch.Tensor)
        assert sample_images.dim() == 4  # batch, height, width, channels
        assert sample_images.shape[-1] == 3  # RGB channels
    
    def test_fixture_device(self, device):
        """Test the device fixture."""
        import torch
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
    
    def test_coverage_import(self):
        """Test that coverage tools are available."""
        try:
            import coverage
            import pytest_cov
            assert True
        except ImportError as e:
            pytest.fail(f"Coverage tools not available: {e}")


@pytest.mark.unit
class TestPytestConfiguration:
    """Test pytest configuration."""
    
    def test_pytest_ini_options(self):
        """Test that pytest.ini options are configured in pyproject.toml."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, "r") as f:
            content = f.read()
        
        # Check key pytest configurations
        assert "[tool.pytest.ini_options]" in content
        assert "testpaths" in content
        assert "--cov" in content
        assert "--cov-report" in content
        assert "markers" in content
    
    def test_coverage_configuration(self):
        """Test that coverage is configured in pyproject.toml."""
        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        with open(pyproject_path, "r") as f:
            content = f.read()
        
        # Check coverage configurations
        assert "[tool.coverage.run]" in content
        assert "[tool.coverage.report]" in content
        assert "fail_under = 80" in content