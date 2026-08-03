"""
utils.py
Shared helper functions used across multiple people's modules.
Keep this file dependency-light so everyone can import it safely.
"""

import json
import os
from datetime import datetime

import joblib


def save_model(model, filepath: str):
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str):
    """Load a trained model from disk."""
    return joblib.load(filepath)


def save_metadata(metadata: dict, filepath: str):
    """
    Save model metadata (params, metrics, timestamp) as JSON.
    Call this every time a new 'best model' is saved so the team
    has a record of what was trained and why it was chosen.
    """
    metadata["saved_at"] = datetime.now().isoformat()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {filepath}")


def load_metadata(filepath: str) -> dict:
    """Load model metadata from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(path: str):
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)
