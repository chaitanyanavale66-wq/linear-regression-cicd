
import numpy as np
import joblib

from src.model import train_model, predict


# Training data
# Area in square feet
X = np.array([
    [500],
    [600],
    [700],
    [800],
    [900],
    [1000],
    [1100],
    [1200],
    [1300],
    [1400]
])

# Price in lakhs
y = np.array([
    25, 30, 35, 40, 45,
    50, 55, 60, 65, 70
])


def main():

    # Train model
    model = train_model(X, y)

    # Generate predictions
    predictions = predict(model, X)

    # Calculate training accuracy (R-squared)
    score = model.score(X, y)

    print("Model training completed")
    print("R2 Score:", score)

    # Save trained model
    joblib.dump(model, "artifacts/house_price_model.pkl")

    print("Model saved successfully")


if __name__ == "__main__":
    main()
