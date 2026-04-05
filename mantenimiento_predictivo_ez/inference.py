import json
import joblib
import numpy as np
import pandas as pd
import __main__
from sklearn.base import BaseEstimator, TransformerMixin

INPUT_FEATURES = ['air_temp', 'process_temp', 'speed', 'torque', 'tool_wear', 'machine_type']


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=INPUT_FEATURES)
        X['power']     = X['torque'] * (X['speed'] * 2 * np.pi / 60)
        X['temp_diff'] = X['process_temp'] - X['air_temp']
        X['strain']    = X['torque'] * X['tool_wear']
        return X


def model_fn(model_dir):
    # pickle serializo la clase como __main__.FeatureEngineer desde el notebook.
    # gunicorn no corre inference.py como __main__, por eso hay que inyectarla
    # en ese modulo antes de que joblib intente deserializar.
    __main__.FeatureEngineer = FeatureEngineer
    return joblib.load(f"{model_dir}/model.joblib")


def input_fn(request_body, content_type='application/json'):
    data = json.loads(request_body)
    if isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)[INPUT_FEATURES]


def predict_fn(input_data, model):
    probs = model.predict_proba(input_data)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return [
        {'failure_prob': round(float(p), 4), 'failure': int(f)}
        for p, f in zip(probs, preds)
    ]


def output_fn(prediction, accept='application/json'):
    body = prediction[0] if len(prediction) == 1 else prediction
    return json.dumps(body), 'application/json'
