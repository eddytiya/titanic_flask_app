from flask import Flask, request, render_template
import h2o
import pandas as pd
import os

# Start H2O
h2o.init()

# Load your saved H2O model
model_path = "models/best_model"
if not os.path.exists(model_path):
    raise Exception("Model folder not found. Make sure 'best_model' exists inside 'models/'")
model = h2o.load_model(model_path)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get form data
        data = {
            "Pclass": [int(request.form["Pclass"])],
            "Sex": [request.form["Sex"]],
            "Age": [float(request.form["Age"])],
            "SibSp": [int(request.form["SibSp"])],
            "Parch": [int(request.form["Parch"])],
            "Fare": [float(request.form["Fare"])],
            "Embarked": [request.form["Embarked"]]
        }
        df = pd.DataFrame(data)
        hf = h2o.H2OFrame(df)
        preds = model.predict(hf)
        prediction = preds.as_data_frame()["predict"][0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
