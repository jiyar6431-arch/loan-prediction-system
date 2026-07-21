from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("loan_prediction_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        no_of_dependents = float(request.form["no_of_dependents"])
        income_annum = float(request.form["income_annum"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        cibil_score = float(request.form["cibil_score"])
        residential_assets_value = float(request.form["residential_assets_value"])
        commercial_assets_value = float(request.form["commercial_assets_value"])
        luxury_assets_value = float(request.form["luxury_assets_value"])
        bank_asset_value = float(request.form["bank_asset_value"])

        features = np.array([[no_of_dependents,
                              income_annum,
                              loan_amount,
                              loan_term,
                              cibil_score,
                              residential_assets_value,
                              commercial_assets_value,
                              luxury_assets_value,
                              bank_asset_value]])

        prediction = model.predict(features)[0]

        if prediction == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
