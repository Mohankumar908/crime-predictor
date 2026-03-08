import joblib

# Load the saved model
model = joblib.load("crime_prediction_model.pkl")

sample_input = [[2024, 6, 18000]]

prediction = model.predict(sample_input)

print("Predicted crime count:", prediction[0])
