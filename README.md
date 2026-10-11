# asd-assessment-using-penguins

This project trains a small machine-learning model and outputs its predictions through running a separate Python file.

- `train.py` trains a classifier on the Palmer Penguins dataset and saves it to `model/penguin_model.joblib`.
- `predict.py` loads the saved model and outputs prediction inputs.

## How to run

In the terminal, run the following:

```bash
pip install -r requirements.txt
python train.py
```
To use the model, run the following:

```bash
python predict.py                    # example provided in code
python predict.py 50 25 150 3100    # alternatively provide model input in terminal command,
```

Output:

```
Input:      {'bill_length_mm': 40, 'bill_depth_mm': 20, 'flipper_length_mm': 100, 'body_mass_g': 3000}
Predicted: Adelie
```
