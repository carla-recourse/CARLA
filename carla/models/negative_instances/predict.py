from sklearn.metrics import accuracy_score


def predict_negative_instances(model, data):
    """Predicts the data target and retrieves the negative instances. (H^-)

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    data : Data
        Dataset used for predictions
    Returns
    -------
    df :  data.api Data() class with negative predicted instances
    """
    df = data.raw
    df["y"] = predict_label(model, data)
    df = df[df["y"] == 0]
    df = df.drop("y", axis="columns")

    return df


def predict_label(model, data, as_prob=False):
    """Predicts the data target

    Assumption: Positive class label is at position 1

    Parameters
    ----------
    name : Tensorflow or PyTorch Model
        Model object retrieved by :func:`load_model`
    data : Data
        Dataset used for predictions
    Returns
    -------
    predictions :  2d numpy array with predictions
    """
    print(f"Predicing label '{data.target}' of {data.name} dataset.")
    features = data.encoded_normalized.drop(data.target, axis=1)
    # Keep correct feature order for prediction
    features = features[model.feature_input_order]
    predictions = model.predict(features)

    if not as_prob:
        predictions = predictions.round()

    acc = accuracy_score(data.raw[data.target], predictions.round())
    print(f"Model accuracy is: {(100* acc).round(2)}%.")

    return predictions
