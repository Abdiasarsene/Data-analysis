# tests/test_model.py
import pytest
from my_ml_package.model import MyModel
import numpy as np

def test_model_training():
    model = MyModel()
    X_train = np.random.rand(100, 4)
    y_train = np.random.randint(2, size=100)
    model.train(X_train, y_train)
    assert len(model.predict(X_train)) == 100
