import grpc
from ml_service_pb2_grpc import MlModuleStub
from ml_service_pb2 import (
    TaskType,
    GetModelsRequest,
    GetHyperParamsRequest,
    PredictOrCreateModelRequest,
    RemoveModelRequest
)

git_repo = 'https://raw.githubusercontent.com/darkydash/mlops_hw_api/main/'

csv_path = git_repo + 'example.csv'

if __name__ == '__main__':
    channel = grpc.insecure_channel("localhost:50051")

    client = MlModuleStub(channel)

    print(client.GetModels(
        GetModelsRequest(task_type=TaskType.CLASSIFICATION)
    ))

    print(client.GetHyperParams(
        GetHyperParamsRequest(model_name='LinearRegression')
    ).hyper_params)

    print(client.GetHyperParams(
        GetHyperParamsRequest(model_name='LogisticRegression')
    ).hyper_params)

    print(client.PredictOrCreateModel(PredictOrCreateModelRequest(
        model_name='LinearRegression',
        target='y',
        hyper_params='{"fit_intercept": true}',
        retrain=True,
        csv_path=csv_path
    )))

    print(client.PredictOrCreateModel(PredictOrCreateModelRequest(
        model_name='LinearRegression',
        target='y',
        csv_path=csv_path
    )))

    print(client.RemoveModel(RemoveModelRequest(
        model_name='LinearRegression'
    )))
