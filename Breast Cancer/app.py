from flask import Flask,render_template,request
import numpy as np 
import pickle 
import pandas as pd 

model = pickle.load(open("./model/model.pkl","rb"))

# flask app 
app = Flask(__name__)

@app.route("/")
def index() :
    return render_template("index.html")

@app.route("/predict",methods=['POST']) 
@app.route("/predict", methods=["POST"])
def predict():
    features = request.form["feature"]

    try:
        features_lst = [float(x.strip()) for x in features.split(",")]
    except ValueError:
        return render_template(
            "index.html",
            message="Invalid input format"
        )

    np_features = np.asarray(features_lst, dtype=np.float32).reshape(1, -1)
    if np_features.shape[1] != 30:
        return render_template(
            "index.html",
            message="Incorrect number of features"
        )

    prediction = model.predict(np_features)[0]

    result = "Cancerous" if prediction == 1 else "Not Cancerous"

    return render_template("index.html", message=result)


if(__name__) == "__main__" :
    app.run(debug=True)