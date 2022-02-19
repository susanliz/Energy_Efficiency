from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('mode_DecisionTreeRegressor.pkl', 'rb'))

kk = model.predict([[0.62, 808.5, 367.5, 220.50, 3.5, 4, 0.4, 5]])
print(kk)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    Relative_Compactness = float(request.form['Relative_Compactness'])
    Surface_Area = float(request.form['Surface_Area'])
    Wall_Area = float(request.form['Wall_Area'])
    Roof_Area = float(request.form['Roof_Area'])
    Overall_Height = float(request.form['Overall_Height'])
    Orientation = int(request.form['Orientation'])
    Glazing_Area = float(request.form['Glazing_Area'])
    Glazing_Area_Distribution = int(request.form['Glazing_Area_Distribution'])

    testdata = model.predict([[Relative_Compactness, Surface_Area, Wall_Area, Roof_Area, Overall_Height, Orientation,
                               Glazing_Area, Glazing_Area_Distribution]])

    cooling = np.round(testdata[0], 2)

    return render_template("result.html", heat=f"Heating: {cooling[0]} cooling: {cooling[1]}")


if __name__ == "__main__":
    app.run(debug=True)
