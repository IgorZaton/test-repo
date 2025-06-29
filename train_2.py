import mlflow
import uuid
import random
import os
from ailine.ailine import run_with_ailine # Import the decorator

# Apply the decorator. Set dvc_track=True to automatically track artifacts.
@run_with_ailine(dvc_track=True)
def train():
    print("Started training run...")
    with mlflow.start_run():
        print("MLflow run started.")
        mlflow.log_param("foo", 123)
        for epoch in range(2):
            mlflow.log_metric("loss", random.random(), step=epoch)
        
        # This artifact will now be automatically detected and added to DVC
        with open("dummy_model.txt", "w") as f:
            f.write("This is a model artifact.")
        mlflow.log_artifact("dummy_model.txt")
    print("Training run finished.")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5001")
    # No need for manual injection anymore, the decorator handles it.
    train()
