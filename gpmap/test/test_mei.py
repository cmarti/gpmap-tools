#!/usr/bin/env python
import sys
import unittest
import numpy as np
import pandas as pd

from os.path import join
from subprocess import check_call
from tempfile import NamedTemporaryFile

from scipy.stats import pearsonr
from scipy.special import comb

from gpmap.src.matrix import quad
from gpmap.src.inference import MinimumEpistasisInterpolator


class MEITests(unittest.TestCase):
    def test_interpolation(self):
        model = MinimumEpistasisInterpolator(P=2)
        X = np.array(['AA', 'AB', 'BA'])
        y = np.array([0, 1, 1.])
        model.set_data(X, y)
        y_pred = model.predict()
        assert(y_pred[-1] == 2.)
        assert(model.cost == 0.)
        
        y_pred = model.predict(smooth=True)
        assert(np.allclose(y_pred, [0, 1, 1, 2]))
        assert(model.cost == 0.)
        
        model = MinimumEpistasisInterpolator(P=2)
        X = np.array(['AAA', 'ABB', 'BAA', 'BBB'])
        y = np.array([1, 0, 0, 1])
        model.set_data(X, y)
        y_pred = model.predict()
        cost1 = model.cost
        assert(np.allclose(y, y_pred[model.obs_idx]))
        assert(cost1 > 0.)
        
        y_pred_smoothed = model.predict(smooth=True)
        assert(np.allclose(y_pred_smoothed[model.pred_idx],
                           y_pred[model.pred_idx]))
        assert(not np.allclose(y, y_pred_smoothed[model.obs_idx]))
        assert(model.cost > 0)
        assert(model.cost < cost1)
        
        
if __name__ == '__main__':
    sys.argv = ['', 'MEITests']
    unittest.main()
