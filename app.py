from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and label encoder
model = joblib.load("weather_model.pkl")
le = joblib.load("label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the form
    data = request.form.to_dict()

    # Convert input data to DataFrame
    input_data = pd.DataFrame([{
        "Temperature": float(data["Temperature"]),
        "Humidity": float(data["Humidity"]),
        "Precipitation (%)": float(data["Precipitation (%)"])
    }])

    # Make prediction
    prediction = model.predict(input_data)
    predicted_weather = le.inverse_transform(prediction)[0]

    return jsonify({"predicted_weather": predicted_weather})


if __name__ == "__main__":
    app.run(debug=True)