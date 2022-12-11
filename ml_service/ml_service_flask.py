import sys
import os
sys.path.append(os.getcwd())

from flask import Flask
from flask_restx import Resource, Api, reqparse
from werkzeug.exceptions import NotFound, BadRequest
from ML_methods.tools import get_model_class
import pandas as pd
import os
import joblib
import json

app = Flask(__name__)
api = Api(app)

model_by_task_type = {
    'classification': ['LogisticRegression', 'KNeighborsClassifier'],
    'regression': ['DecisionTreeRegressor', 'LinearRegression']
}


@api.route('/get_models/<string:task_type>')
class GetModelsRest(Resource):
    def get(self, task_type):
        if task_type not in model_by_task_type.keys():
            raise NotFound('No such types models exist')

        return {
            'model_names': model_by_task_type[task_type]
        }


@api.route('/get_hyper_params/<string:model_name>')
class GetHyperParamsRest(Resource):
    def get(self, model_name):
        model_class = get_model_class(model_name)
        if model_class is None:
            raise NotFound('No such model exists')

        return {
            'hyper_params': list(model_class().get_params().keys())
        }


predict_create_parser = reqparse.RequestParser()
predict_create_parser.add_argument('retrain', type=bool, location='form')
predict_create_parser.add_argument('target', type=str, location='form')
predict_create_parser.add_argument('hyper_params', type=str, location='form')
predict_create_parser.add_argument('csv_path', type=str, location='form')
predict_create_parser.add_argument('model_name', type=str, location='form')


@api.route('/predict_or_create_model/')
@api.expect(predict_create_parser)
class PredictOrCreateModelRest(Resource):
    def post(self):
        args = predict_create_parser.parse_args()
        retrain = args.get('retrain')
        target = args.get('target')
        hyper_params = args.get('hyper_params')
        csv_path = args.get('csv_path')
        model_name = args.get('model_name')

        model_class = get_model_class(model_name)

        if model_class is None:
            raise NotFound('No such model exists')

        if csv_path == '':
            raise BadRequest('No data provided')

        data = pd.read_csv(csv_path)

        if not retrain and \
                f"{model_name}.joblib" in os.listdir('./saved_models'):
            if target != '':
                data.drop(target, axis=1, inplace=True)

            model = joblib.load(f"./saved_models/{model_name}.joblib")
            pred = model.predict(data).flatten().astype('float64').tolist()

            return {
                'status': 'Predicted',
                'predict': pred
            }

        if f"{model_name}.joblib" in os.listdir('./saved_models'):
            os.unlink(f"./saved_models/{model_name}.joblib")

        if target == '':
            return BadRequest('No target column name provided')

        y = data[target]
        X = data.drop(target, axis=1)

        if hyper_params:
            hyper_params = json.loads(hyper_params)
        else:
            hyper_params = {}

        model = model_class(**hyper_params)

        model.fit(X, y)

        joblib.dump(model, f"./saved_models/{model_name}.joblib")

        return {
            'status': 'Model has been trained and saved'
        }


@api.route('/remove_model/<string:model_name>')
class RemoveModel(Resource):
    def delete(self, model_name):
        if f"{model_name}.joblib" in os.listdir('./saved_models'):
            os.unlink(f"./saved_models/{model_name}.joblib")

            return {
                'status': 'Model removed successfully'
            }

        return {
            'status': 'Model does not exist, nothing to do'
        }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
