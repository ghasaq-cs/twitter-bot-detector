
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load(r'C:\Users\ghasq\bot_model.pkl')

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Bot Detector</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
        input { width: 100%; padding: 8px; margin: 5px 0 15px 0; }
        button { background: #1da1f2; color: white; padding: 10px 20px; border: none; cursor: pointer; width: 100%; }
        #result { margin-top: 20px; padding: 15px; font-size: 20px; text-align: center; }
    </style>
</head>
<body>
    <h2>🤖 Twitter Bot Detector</h2>
    <label>Followers Count</label>
    <input type="number" id="followers" placeholder="e.g. 1500">
    <label>Friends Count</label>
    <input type="number" id="friends" placeholder="e.g. 300">
    <label>Favourites Count</label>
    <input type="number" id="favourites" placeholder="e.g. 200">
    <label>Statuses Count</label>
    <input type="number" id="statuses" placeholder="e.g. 500">
    <label>Average Tweets Per Day</label>
    <input type="number" id="tweets_per_day" placeholder="e.g. 5" step="0.1">
    <label>Account Age (days)</label>
    <input type="number" id="age" placeholder="e.g. 365">
    <button onclick="predict()">Analyze Account</button>
    <div id="result"></div>
    <script>
        async function predict() {
            const data = {
                followers: document.getElementById('followers').value,
                friends: document.getElementById('friends').value,
                favourites: document.getElementById('favourites').value,
                statuses: document.getElementById('statuses').value,
                tweets_per_day: document.getElementById('tweets_per_day').value,
                age: document.getElementById('age').value
            };
            const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)});
            const result = await res.json();
            const div = document.getElementById('result');
            if(result.prediction === 1) {
                div.innerHTML = '🤖 BOT DETECTED';
                div.style.background = '#ffcccc';
            } else {
                div.innerHTML = '✅ HUMAN';
                div.style.background = '#ccffcc';
            }
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        float(data['followers']),
        float(data['friends']),
        float(data['favourites']),
        float(data['statuses']),
        float(data['tweets_per_day']),
        float(data['age'])
    ]])
    prediction = model.predict(features)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)