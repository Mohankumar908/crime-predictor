from django.shortcuts import render
import pandas as pd
import joblib

# Create your views here.
model = joblib.load("model/crime_prediction_model.pkl")

def home(request):
    prediction = None
    if request.method == "POST":

        year = int(request.POST['year'])
        month = int(request.POST['month'])
        prev_crime = int(request.POST['prev_crime'])

        data = pd.DataFrame({
            'Year' : [year],
            'Month' : [month],
            'Prev_Month_Crime' : [prev_crime]
        })

        result = model.predict(data)

        prediction = int(result[0])

    return render(request, "index.html",{"prediction": prediction})