
import numpy as np

from src.model import train_model, predict


def test_model_training():

    X = np.array([[500], [600], [700], [800]])
    y = np.array([25, 30, 35, 40])

    model = train_model(X, y)

    assert model is not None


def test_model_prediction():

    X = np.array([[500], [600], [700], [800]])
    y = np.array([25, 30, 35, 40])

    model = train_model(X, y)

    prediction = predict(model, [[1000]])

    assert prediction[0] == 50


def test_model_score():

    X = np.array([[500], [600], [700], [800]])
    y = np.array([25, 30, 35, 40])

    model = train_model(X, y)

    assert model.score(X, y) > 0.95
