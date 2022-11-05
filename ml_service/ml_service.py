import grpc

from ml_service_pb2 import (
    TaskType,
    GetModelsResponse,
    GetHyperParamsResponse,
    PredictOrCreateModelResponse,
    RemoveModelResponse
)

from ML_methods.tools import get_model_class

import ml_service_pb2_grpc

from concurrent import futures

import joblib

import json

import os

import pandas as pd

model_by_task_type = {
    TaskType.CLASSIFICATION: ['LogisticRegression', 'KNeighborsClassifier'],
    TaskType.REGRESSION: ['DecisionTreeRegressor', 'LinearRegression']
}


class MlService(
    ml_service_pb2_grpc.MlModuleServicer
):
    def GetModels(self, request, context):
        if request.task_type not in model_by_task_type:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                "No such types models exist"
            )

        return GetModelsResponse(
            model_names=model_by_task_type[request.task_type]
        )

    def GetHyperParams(self, request, context):
        model_class = get_model_class(request.model_name)

        if model_class is None:
            return context.abort(
                grpc.StatusCode.NOT_FOUND,
                "No such model exists"
            )

        return GetHyperParamsResponse(
            hyper_params=list(model_class().get_params().keys())
        )

    def PredictOrCreateModel(self, request, context):

        if request.csv_path == '':
            return context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "No data provided"
            )

        data = pd.read_csv(request.csv_path)

        # Predict case
        if not request.retrain and \
                f"{request.model_name}.joblib" in os.listdir('./saved_models'):

            if request.target != '':
                data.drop(request.target, axis=1, inplace=True)

            model = joblib.load(f"./saved_models/{request.model_name}.joblib")
            pred = model.predict(data).flatten().astype('float64').tolist()

            return PredictOrCreateModelResponse(
                status="Predicted",
                predict=pred
            )

        # If model exists, retrain case
        if f"{request.model_name}.joblib" in os.listdir('./saved_models'):
            os.unlink(f"./saved_models/{request.model_name}.joblib")

        # Train model otherwise
        if request.target == '':
            return context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "No target column name provided"
            )

        y = data[request.target]
        X = data.drop(request.target, axis=1)

        if request.hyper_params != '':
            hyper_params = json.loads(request.hyper_params)
        else:
            hyper_params = {}

        model = get_model_class(request.model_name)(**hyper_params)
        model.fit(X, y)

        joblib.dump(model, f"./saved_models/{request.model_name}.joblib")

        return PredictOrCreateModelResponse(
            status="Model has been trained and saved"
        )

    def RemoveModel(self, request, context):
        if f"{request.model_name}.joblib" in os.listdir('./saved_models'):
            os.unlink(f"./saved_models/{request.model_name}.joblib")

            return RemoveModelResponse(status="Model removed successfully")

        return RemoveModelResponse(
            status="Model does not exist, nothing to do"
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    ml_service_pb2_grpc.add_MlModuleServicer_to_server(
        MlService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
