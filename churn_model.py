import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ChurnModel:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.feature_names = []

    def load_model(self, model_path="customer_churn_model.pkl", encoders_path="encoders.pkl"):
        try:
            self.model = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        try:
            self.encoders = joblib.load(encoders_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load encoders from {encoders_path}: {e}")
        
        if hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            self.feature_names = []

    def preprocess(self, input_dict):
        df = pd.DataFrame([input_dict])

        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    # Ensure encoder is fitted
                    if not hasattr(encoder, 'classes_'):
                        raise ValueError(f"Encoder for column '{col}' is not fitted.")

                    # Replace unseen values with 'Unknown'
                    df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')

                    # Add 'Unknown' to classes if missing
                    if 'Unknown' not in encoder.classes_:
                        encoder.classes_ = np.append(encoder.classes_, 'Unknown')

                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    raise ValueError(f"Error encoding column '{col}': {e}")
        return df

    def predict(self, input_dict):
        if self.model is None or not self.encoders:
            raise RuntimeError("You must call load_model() before predict().")

        df = self.preprocess(input_dict)
        pred = self.model.predict(df)[0]
        proba = self.model.predict_proba(df)[0]
        return pred, proba
      