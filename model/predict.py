
import pickle
import pandas as pd
from pathlib import Path

# Import the ML model
model_path = Path(__file__).parent / "model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

# MLFlow
MODEL_VERSION = "1.0.0"

# Get class labels
class_labels = model.classes_.tolist()


def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    # Predict the class
    predicted_class = model.predict(df)[0]

    # Get probabilities for all classes
    probabilities = model.predict_proba(df)[0]

    confidence = max(probabilities)

    # Create mapping of class names and probabilities
    class_probs = dict(
        zip(
            class_labels,
            map(lambda p: round(float(p), 4), probabilities)
        )
    )

    return {
        "predicted_category": predicted_class.item() if hasattr(predicted_class, "item") else predicted_class,
        "confidence": round(float(confidence), 4),
        "class_probabilities": class_probs
    }
