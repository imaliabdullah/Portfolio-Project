import flask
import json
import numpy as np
from joblib import dump, load
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	return flask.render_template('index.html')

@app.route("/predict", methods=['POST'])
def make_predictions():
        if request.method =='POST':
                age = request.form.get('age')
                sex = request.form.get('sex')
                cp = request.form.get('chest-pain')
                trestbps = request.form.get('rest-bp')
                chol = request.form.get('serum-cholestoral')
                fbs = request.form.get('fasting-blood')
                restecg = request.form.get('rest-ecg')
                thalach = request.form.get('max-heart-rate')
                exang = request.form.get('ex-induced')
                oldpeak = request.form.get('old-peak')
                slope = request.form.get('slope')
                ca = request.form.get('num-vessels')
                thal = request.form.get('thal')
                X = np.array([age,sex,cp,trestbps,chol,fbs,
                              restecg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1)
                prediction = model.predict(X)

                resultText = json.dumps(prediction.tolist(),sort_keys = False)
                resultTextObject = json.loads(resultText)
                # This json.dumps will convert 'prediction' to JSON

                return render_template('predictPage.html',response=resultTextObject[0])
                # 'response' is a Jinja2 variable embedded in predictPage.html

@app.route('/api')
def hello():
    response = {'MESSAGE':'Welcome to the new API route'}
    return jsonify(response)

if __name__ == '__main__':
    model = load('C:\Ali Abdullah\Github Repository\others\Data-Science\DS-Project\Heat Disease\model_rf.pkl')
    app.run(host='0.0.0.0', port=8000, debug=True)