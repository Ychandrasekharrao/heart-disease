"""
Model prediction functions.
"""


def predict(model, X):
    """
    Make predictions using the trained model.

    Args:
        model: Trained model object
        X: Features to make predictions on

    Returns:
        Array of predictions
    """
    return model.predict(X)


def predict_proba(model, X):
    """
    Get probability predictions using the trained model.

    Args:
        model: Trained model object
        X: Features to make predictions on

    Returns:
        Array of probability predictions
    """
    return model.predict_proba(X)
