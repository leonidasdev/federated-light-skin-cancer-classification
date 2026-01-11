"""
Data Tests
==========

Unit tests for dataset loaders.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_train_transforms, get_val_transforms


class TestTransforms:
    """Tests for data transforms."""
    
    def test_train_transforms_output_shape(self):
        """Test training transforms produce correct output shape."""
        from PIL import Image
        import numpy as np
        
        transform = get_train_transforms(img_size=224)
        
        # Create dummy image
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        
        output = transform(dummy_img)
        assert output.shape == (3, 224, 224)
    
    def test_val_transforms_output_shape(self):
        """Test validation transforms produce correct output shape."""
        from PIL import Image
        import numpy as np
        
        transform = get_val_transforms(img_size=224)
        
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        
        output = transform(dummy_img)
        assert output.shape == (3, 224, 224)
    
    def test_transforms_normalization(self):
        """Test that transforms apply normalization."""
        from PIL import Image
        import numpy as np
        
        transform = get_val_transforms(img_size=224)
        
        # All white image
        white_img = Image.fromarray(
            np.ones((224, 224, 3), dtype=np.uint8) * 255
        )
        
        output = transform(white_img)
        # After normalization, values should not be in [0, 1]
        assert output.max() > 1.0 or output.min() < 0.0


class TestDatasets:
    """Tests for dataset classes."""
    
    @pytest.mark.skip(reason="Requires actual data files")
    def test_ham10000_loading(self):
        """Test HAM10000 dataset loading."""
        pass
    
    @pytest.mark.skip(reason="Requires actual data files")
    def test_isic2018_loading(self):
        """Test ISIC 2018 dataset loading."""
        pass


class TestFederatedPartitioner:
    """Tests for federated dataset partitioner."""
    
    @pytest.mark.skip(reason="Partitioner not yet implemented")
    def test_iid_partition(self):
        """Test IID partitioning produces balanced splits."""
        pass
    
    @pytest.mark.skip(reason="Partitioner not yet implemented")
    def test_dirichlet_partition(self):
        """Test Dirichlet partitioning produces heterogeneous splits."""
        pass
