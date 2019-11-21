from flask import Flask, jsonify, request
import pickle
import pandas as pd
import json

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def apicall():
    try:
        test_json = request.get_json(force=True)
        test = pd.read_json(json.dumps(test_json), orient='records')  # reconds格式，转化为dataframe数据格式
        loan_ids = test['USR_NUM_ID']

    except Exception as e:
        raise e

    clf = 'churn_xgb.model'

    if test.empty:
        return (bad_request())
    else:
        print("Loading the model...")
        loaded_model = None
        with open('/Users/gengpeng/' + clf, 'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict_proba(test)[:, 1]
        prediction_series = list(pd.Series(predictions))
        res = dict(zip(loan_ids, prediction_series))
        responses = jsonify(predictions=res)
        responses.status_code = 200

        return (responses)


@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp


if __name__ == '__main__':
    app.run()

