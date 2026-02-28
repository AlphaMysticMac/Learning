# MLflow – Learning Notes

## Overview
MLflow is an open-source MLOps platform used to manage and track the machine learning lifecycle. It helps data scientists record experiments, compare model performance, and reproduce results efficiently.

---

## Key Concepts

### Experiment
- Logical container for grouping related ML work.
- Represents a single ML problem or project.
- Example: *Customer Churn Prediction* or *Fraud Detection Model*.
- Stores multiple training attempts (runs).

### Run
- A single execution of model training code.
- Automatically created whenever training starts.
- Tracks all information related to that execution.

---

## What MLflow Logs

- **Parameters:** Hyperparameters such as learning rate, depth, batch size.
- **Metrics:** Model performance values like accuracy, F1-score, loss.
- **Artifacts:** Output files including models, plots, datasets, logs.
- **Source Info:** Code version and execution metadata for reproducibility.

---

## Features
- Experiment tracking and comparison via UI dashboard.
- Framework-agnostic (Scikit-learn, PyTorch, TensorFlow, XGBoost).
- Model versioning using Model Registry.
- Easy collaboration across teams.
- Integration with cloud storage and deployment pipelines.

---

## Why Use MLflow
- Ensures experiment reproducibility.
- Prevents loss of model configurations.
- Simplifies model selection and auditing.
- Supports production-ready ML workflows.
