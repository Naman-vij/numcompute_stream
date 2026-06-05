"""Unit tests for stats.py"""

import numpy as np
import pytest
from numcompute_stream.stats import StreamingMean, StreamingVariance


class TestStreamingMean:
    
    def test_single_value(self):
        mean = StreamingMean()
        mean.update(5.0)
        assert mean.result() == 5.0
    
    def test_multiple_values(self):
        mean = StreamingMean()
        mean.update([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mean.result() == 3.0
    
    def test_incremental_update(self):
        mean = StreamingMean()
        mean.update([1.0, 2.0])
        mean.update([3.0, 4.0])
        mean.update([5.0])
        assert mean.result() == 3.0
    
    def test_with_nan(self):
        mean = StreamingMean()
        mean.update([1.0, np.nan, 3.0])
        assert mean.result() == 2.0
    
    def test_reset(self):
        mean = StreamingMean()
        mean.update([1.0, 2.0])
        mean.reset()
        assert mean.result() == 0.0


class TestStreamingVariance:
    
    def test_single_value(self):
        var = StreamingVariance()
        var.update(5.0)
        assert var.result() == 0.0
    
    def test_two_values(self):
        var = StreamingVariance(ddof=1)
        var.update([1.0, 3.0])
        expected = ((1-2)**2 + (3-2)**2) / 1
        assert abs(var.result() - expected) < 1e-10
    
    def test_incremental_update(self):
        var1 = StreamingVariance()
        var1.update([1.0, 2.0, 3.0, 4.0, 5.0])
        
        var2 = StreamingVariance()
        var2.update([1.0, 2.0])
        var2.update([3.0, 4.0, 5.0])
        
        assert abs(var1.result() - var2.result()) < 1e-10
    
    def test_std(self):
        var = StreamingVariance()
        var.update([1.0, 2.0, 3.0])
        std = var.std()
        assert std >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])