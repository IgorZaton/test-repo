import mlflow
import os
import time
import random

def train():
    # Retrieve run ID tag passed by wrapper
    ailine_run_id = os.getenv("AILINE_RUN_ID")

    # Start an MLflow run
    with mlflow.start_run() as run:
        # Set identifying tag so the wrapper can find this run
        if ailine_run_id:
            mlflow.set_tag("AILINE_RUN_ID", ailine_run_id)

        # Simulate logging parameters and metrics
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("batch_size", 32)

        # Simulate training metrics
        for epoch in range(5):
            acc = 0.8 + random.uniform(0, 0.2)
            loss = 1.0 / (epoch + 1) + random.uniform(0, 0.1)
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("loss", loss, step=epoch)
            time.sleep(0.1)  # simulate work

        # Optionally, log an artifact
        with open("dummy_model.txt", "w") as f:
            f.write("Pretend this is a model artifact.")
        mlflow.log_artifact("dummy_model.txt")

if __name__ == "__main__":
    train()
