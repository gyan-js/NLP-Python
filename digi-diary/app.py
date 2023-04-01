from flask import Flask, render_template, request, jsonify
from model_prediction import *
from dotenv import load_dotenv
import os
app = Flask(__name__)

text=""
predicted_emotion=""
predicted_emotion_img_url=""

load_dotenv()

@app.route("/")
def home():
    entries = show_entry()
    return render_template("index.html", entries=entries)
    

@app.route("/predict-emotion", methods=["POST"])
def predict_emotion():
    input_text = request.json.get("text")
    if not input_text:
        return jsonify({
            "status": "error",
            "message": "Please enter some text to predict emotion!"
        }), 400
    else:
        predicted_emotion, predicted_emotion_img_url = predict(input_text)                         
        return jsonify({
            "data": {
                "predicted_emotion": predicted_emotion,
                "predicted_emotion_img_url": predicted_emotion_img_url
            },
            "status": "success"
        }), 200
        
@app.route("/save-entry", methods=["POST"])

def save_entry():
    data_entry_path = os.getenv("PATH_TO_DATA_ENTRY")
    date = request.json.get("date")
    emotion = request.json.get("emotion")
    save_text = request.json.get("text")
    emotion_url = request.json.get("emotion_url")
    save_text = save_text.replace("\n", "")

    entry = f'"{date}","{save_text}","{emotion}","{emotion_url}"\n'


    with open(data_entry_path, 'a') as f:
        f.write(entry)
    return jsonify("Success")

                
if __name__ == "__main__":
    app.run(debug=True)

