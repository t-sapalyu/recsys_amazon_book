import random
import pickle
import os
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
import mlflow
import mlflow.pyfunc

USE_TEMP_MODEL = False


class EnergyModel(ABC):
    version: str

    @abstractmethod
    def predict(self, features: dict) -> float: ...


class TempModel(EnergyModel):
    def __init__(self):
        self.version = "Mock-v0.1"

    def predict(self, features: dict) -> float:
        return round(random.uniform(1000.0, 5000.0), 1)


class RealModel(EnergyModel):
    def __init__(self):
        model_paths = ["baseline_model.pkl", "model_service/baseline_model.pkl"]
        self.model = None
        found_path = None

        for path in model_paths:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.model = pickle.load(f)
                found_path = path
                break

        if self.model is None:
            raise FileNotFoundError("Could not locate baseline_model.pkl!")

        mtime = datetime.fromtimestamp(os.path.getmtime(found_path)).strftime("%Y%m%d-%H%M%S")
        self.version = f"Local-Baseline-{mtime}"
        

    def predict(self, features: dict) -> float:
        date_obj = features['date']

        # Calculate derived temporal features
        day_of_year = date_obj.timetuple().tm_yday
        month_name = date_obj.strftime("%B")
        day_name = date_obj.strftime("%A")

        # Determine the Season
        m = date_obj.month
        if m in [12, 1, 2]:
            season = "Winter"
        elif m in [3, 4, 5]:
            season = "Spring"
        elif m in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"

        # Build the dataframe with all 7 features for the Random Forest
        input_df = pd.DataFrame([{
            'Start_Hour': features['start_hour'],
            'End_Hour': features['end_hour'],
            'Source': features['energy_source'],
            'Month_Name': month_name,
            'Season': season,
            'Day_of_Year': day_of_year,
            'Day_Name': day_name
        }])

        prediction = self.model.predict(input_df)[0]

        if prediction < 0:
            prediction = 0.0

        return round(float(prediction), 1)


class MLflowChampionModel(EnergyModel):
    """
    Loads the promoted @champion model directly from the MLflow Model Registry.
    """
    def __init__(self):
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = "models:/RenewableEnergyModel@champion"
        print(f"Attempting to load MLflow model from {model_uri}...")

        self.model = mlflow.pyfunc.load_model(model_uri)

        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias("RenewableEnergyModel", "champion")
        self.version = f"MLflow-Prod-v{mv.version}"

    def predict(self, features: dict) -> float:
        date_obj = features['date']

        day_of_year = date_obj.timetuple().tm_yday
        month_name = date_obj.strftime("%B")
        day_name = date_obj.strftime("%A")

        m = date_obj.month
        if m in [12, 1, 2]:
            season = "Winter"
        elif m in [3, 4, 5]:
            season = "Spring"
        elif m in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"

        input_df = pd.DataFrame([{
            'Start_Hour': features['start_hour'],
            'End_Hour': features['end_hour'],
            'Source': features['energy_source'],
            'Month_Name': month_name,
            'Season': season,
            'Day_of_Year': day_of_year,
            'Day_Name': day_name
        }])

        prediction = self.model.predict(input_df)[0]

        if prediction < 0:
            prediction = 0.0

        return round(float(prediction), 1)


def load_model():
    """
    Attempts to load the MLflow Champion. Falls back to the local baseline
    if MLflow is empty (e.g., before the first Airflow training run completes).
    """
    if USE_TEMP_MODEL:
        return TempModel()

    try:
        return MLflowChampionModel()
    except Exception as e:
        print(f"MLflow champion not found. Falling back to local .pkl. Reason: {e}")
        return RealModel()
