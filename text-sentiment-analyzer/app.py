from flask import Flask, render_template, url_for, request, jsonify
from text_sentiment_prediction import *

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/predict-emotion', methods=["POST"])
def predict_emotion():
    
    input_text = request.json.get("text")
   
    
    if not input_text:
        respose = {
            "status": "error",
            "message": "Please enter a text"
        }
        return jsonify(respose)
    else:
        predicted_emotion, predicted_emotion_img_url = predict(input_text)
        respose = {
            "status": "success",
            "data": {
                "predicted_emotion": predicted_emotion,
                "predicted_emotion_img_url": predicted_emotion_img_url
            }
        }
        return jsonify(respose)
        
       
app.run(debug=True)



    