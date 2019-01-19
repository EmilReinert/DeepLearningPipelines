import unittest
import torch
import numpy as np
import os
import sys


import modules.models.logistic_regression as lr

lo_reg_config = {"epochs": 400, "learning_rate": 0.02}
_lr = lr.LogisticRegression(lo_reg_config)
X_train = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
Y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])


class TestPublicFunctions(unittest.TestCase):
    # checks if calculations are correct
    def test_asset_sigmoid_calc(self):
        cal = lr.sigmoid(2)
        self.assertAlmostEqual(cal, 0.8807970779778823)


class TestLoRegInit(unittest.TestCase):
    # checks if default init works correctly
    def test_assert_default_init(self):
        de_lr = lr.LogisticRegression()
        self.assertTrue(de_lr.epochs == 50 and de_lr.learning_rate ==
                        0.01)

    # checks if nondefault init works
    def test_assert_non_default_init(self):
        self.assertTrue(_lr.epochs == 400 and _lr.learning_rate ==
                        0.02)


class TestLoRegTraining(unittest.TestCase):
    # checks if training input is correct
    def test_asset_training_input(self):
        _lr.train(X_train, Y_train)


class TestlogTrainTesting(unittest.TestCase):
    # checks if models accuracy makes sense
    def test_asset_testaccuracy(self):
        _lr.train(X_train, Y_train)
        prediction1 = _lr.test(0.1)

        self.assertTrue(prediction1 == 1)  # and prediction2 == 1)


if __name__ == "__main__":
    unittest.main()
