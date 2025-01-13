from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('amazon_sentiment_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    sentiment = model.predict([review])[0]
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(port=5000)
