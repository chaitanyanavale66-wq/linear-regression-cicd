
from sklearn.linear_model import LinearRegression


def train_model(X, y):
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict(model, X):
    """
    Generate predictions using the trained model.
    """
    return model.predict(X)
