"""
Runs the prediction model.
Run: either 
    -"python predict.py"  with the code example or
    -"python predict.py 50 25 150 3100" with providing an input directly via terminal.
"""
import joblib
import pandas as pd
import sys

# Load the trained model
bundle = joblib.load("model/penguin_model.joblib")
model = bundle["model"]
class_names = bundle["class_names"]

def predict(data):

    features = {
            "bill_length_mm": [data["bill_length_mm"]],
            "bill_depth_mm": [data["bill_depth_mm"]],
            "flipper_length_mm" : [data["flipper_length_mm"]],
            "body_mass_g": [data["body_mass_g"]],
    }
    prediction = model.predict(pd.DataFrame.from_dict(features))[0]
    return {"species": prediction}


if __name__ == "__main__":
    if len(sys.argv) == 5:
        values = []
        for val in sys.argv[1:]:
            values.append(float(val))
    else:
        values = [40, 20, 100, 3000]

    penguin = {
        "bill_length_mm": values[0],
        "bill_depth_mm": values[1],
        "flipper_length_mm": values[2],
        "body_mass_g": values[3],
    }

    results = predict(penguin)
    print("Input:     ", penguin)
    print("Prediction:     ", results['species'])