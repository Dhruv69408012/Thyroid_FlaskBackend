from flask import Flask, request, jsonify
import xgboost as xgb
import pickle
from flask_cors import CORS
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

encoding = {
    0: 'PRIMARY HYPORHYROID',
    1: 'SECONDARY HYPOTHYROID',
    2: 'COMPENSATED HYPOTHYROID',
    3: 'Hypothyroid',
    4: 'Hyperthyroid',
    5: 'NEGATIVE'
}

def cleaning(input_data):
    float_features = ["TSH", "TT4", "FTI", "T3"]
    for feature in float_features:
        if feature in input_data and input_data[feature] != "":
            input_data[feature] = float(input_data[feature])

    if input_data["TSH"] == "":
        input_data["TSH"] = 2.875
    if input_data["T3"] == "":
        input_data["T3"] = 1.25
    if input_data["TT4"] == "":
        input_data["TT4"] = 105
    if input_data["FTI"] == "":
        input_data["FTI"] = 97.5

    return input_data

@app.route('/')
def hello():
    return "<p>Hello<p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        client=MongoClient("mongodb+srv://dhruvvayugundla:DDD123ddd@cluster1.j4w98fr.mongodb.net/Thyroid?retryWrites=true&w=majority")
        db=client.Thyroid
        collection=db.users
        data = request.get_json()
        input_data = cleaning(data)
        input_list = [
            input_data["age"], input_data["gender"], input_data["on thyroxine"],
            input_data["on antithyroid medication"], input_data["sick"],
            input_data["pregnant"], input_data["thyroid surgery"],
            input_data["I131 treatment"], input_data["lithium"],
            input_data["goitre"], input_data["tumor"], input_data["hypopituitary"],
            input_data["psych"], input_data["TSH"], input_data["T3"],
            input_data["TT4"], input_data["FTI"]
        ]

        with open("xgmodel.pkl", "rb") as model_file:
            model = pickle.load(model_file)

        y_pred = model.predict([input_list])

        resulting = [encoding[i] for i in y_pred]
        result = {"condition": resulting[0]}
        print(input_data["uname"])
        collection.update_one({"uname":input_data["uname"]},{"$set":{"condition":resulting[0]}})
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run()
