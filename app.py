from flask import Flask, render_template, request
import pandas as pd
from h2o import mojo_predict

MOJO_PATH = "models/best_model.zip"

app = Flask(__name__)
mojo_model = mojo_predict.MojoPredictor(MOJO_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input
        df = pd.DataFrame([{
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked']
        }])
        
        # Predict
        preds = mojo_model.predict(df)
        prediction = preds[0]
        
        return render_template('index.html', prediction_text=f"Predicted Survived: {prediction}")
    
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
