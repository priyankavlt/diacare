from flask import Flask, render_template, request

from diacare.predict import predict

app = Flask(__name__)

# form field -> (model feature, label, unit/hint, min, max, step)
FIELDS = [
    ("pregnancies", "Pregnancies", "Number of pregnancies", "e.g. 0", 0, 20, 1),
    ("glucose", "Glucose", "Glucose (mg/dL)", "e.g. 120", 0, 300, 1),
    ("bloodpressure", "BloodPressure", "Diastolic blood pressure (mmHg)", "e.g. 70", 0, 200, 1),
    ("skinthickness", "SkinThickness", "Triceps skin thickness (mm)", "e.g. 20", 0, 100, 1),
    ("insulin", "Insulin", "2-hour serum insulin (µU/mL)", "e.g. 80", 0, 900, 1),
    ("bmi", "BMI", "Body mass index (kg/m²)", "e.g. 25.0", 0, 80, 0.1),
    ("dpf", "DPF", "Diabetes pedigree function", "e.g. 0.5", 0, 3, 0.001),
    ("age", "Age", "Age (years)", "e.g. 35", 1, 120, 1),
]


@app.route("/")
def home():
    return render_template("index.html", fields=FIELDS, values={}, error=None)


@app.route("/predict", methods=["POST"])
def predict_route():
    patient = {}
    for name, feature, label, _, lo, hi, _ in FIELDS:
        try:
            value = float(request.form[name])
        except (KeyError, ValueError):
            return _form_error(f"{label}: please enter a number.")
        if not lo <= value <= hi:
            return _form_error(f"{label}: must be between {lo} and {hi}.")
        patient[feature] = value

    positive, prob = predict(patient)
    return render_template("result.html", prediction=int(positive), probability=prob)


def _form_error(message):
    return render_template("index.html", fields=FIELDS, values=request.form, error=message), 400


if __name__ == "__main__":
    app.run(debug=True)
