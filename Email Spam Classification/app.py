from flask import Flask, render_template, request
import pickle
from transform_text import text_formation

app = Flask(__name__)
tfidf = pickle.load(open("model/vectorizer.pkl", 'rb'))
model = pickle.load(open("model/model.pkl", 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['post'])
def predict():
    email = request.form.get('email')
    print(email)
    # 1. preprocess
    transformed_sms = text_formation(email)
    # 2. vectorize 
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]

    result = 1 if result == 1 else -1
    return render_template('index.html', response=result)


if __name__ == "__main__":
    app.run(debug=True)
