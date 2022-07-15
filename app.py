from flask import Flask, jsonify,request,render_template
import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf
app = Flask(__name__)


__words = None
__data = None
__model = None
__classes = None

@app.route("/",methods=['GET','POST'])
def index():
    return render_template("app.html")
    

@app.route("/get_response",methods=['GET','POST'])
def get_response():
    with open("./artifacts/words.txt",'r') as f:
        __words = f.read()
    with open("./artifacts/file.json",'r') as f:
         __data = json.loads(f.read())

    with open("./artifacts/classes.txt",'r') as f:
         __classes = f.read()  
            
    __model = tf.keras.models.load_model("./model/1")
    __words = __words.split(",")
    __classes = __classes.split(",")
    print("***********",__words)
    if(type(__words) != "NoneType"):
    # response.headers.add('Access-Control-Allow-Origin','*')
        user_response = request.form['user_text']
        bot_response = get_bot_response(user_response,__words,__data,__classes,__model)
        response = jsonify({
            'bot_response':bot_response
        })
        response.headers.add('Access-Control-Allow-Origin','*')
    return response

def get_bot_response(input,words,data,classes,model):
    global __words = words
    global __data = data
    global __classes = classes
    global __model = model
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(input.lower())
    bow_list = []
    for word in __words:
        bow_list.append(1) if word.lower() in text else bow_list.append(0)
    result = __model.predict(np.array([bow_list]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    assign = y_pred
    y_pred.sort(key=lambda x: x[1], reverse=True) 
    y_pred = y_pred[:1]
    lables = __classes[y_pred[0][0]]
    for intent in __data["intents"]:
        if intent['tag'] == lables:
            result = random.choice(intent["responses"])
            break
    return result


if __name__ =="__main__":
    # load_artifacts()
    app.run()
