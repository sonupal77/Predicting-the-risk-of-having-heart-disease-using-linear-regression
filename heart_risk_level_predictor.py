from flask import Flask, render_template, request
import numpy as np
import joblib

model = joblib.load('regression-model-heart-risk.sav')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/getresults', methods=['POST'])
def getresults():
    form_data = request.form
    name = form_data['name']
    gender = float(form_data['gender'])
    age = float(form_data['age'])
    tc = float(form_data['tc'])
    hdl = float(form_data['hdl'])
    smoke = float(form_data['smoke'])
    bpm = float(form_data['bpm'])
    diab = float(form_data['diab'])

    test_data = np.array([gender,age,tc,hdl,smoke,bpm,diab]).reshape(1,7)
    prediction = model.predict(test_data)
    result_dict = {'name':name, 'risk':round(prediction[0],2)}
    return render_template('result.html', results=result_dict)

app.run(debug=True)