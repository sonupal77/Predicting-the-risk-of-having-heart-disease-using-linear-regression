from flask import Flask, render_template, request
import numpy as np
import joblib

model = joblib.load('regression-model-heart-risk.sav')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')


app.run(debug=True)