def get_model_class(model: str):
    match model.lower():
        case 'linearregression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression
        case 'decisiontreeregressor':
            from sklearn.tree import DecisionTreeRegressor
            return DecisionTreeRegressor
        case 'logisticregression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression
        case 'KNeighborsClassifier':
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier
        case _:
            return None
