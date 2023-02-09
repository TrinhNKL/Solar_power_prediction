from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
import pandas as pd
import numpy as np

def prediction_model(X_train, y_train, X_test):
    model = RandomForestRegressor(n_estimators = 100)
    model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)

    return model
    

def feature_important(model,X_train, y_train):
    feature_importances = model.feature_importances_

    X_train_opt = X_train.copy()
    removed_columns = pd.DataFrame()
    models_features = []
    r2s_opt = []

    for i in range(0,5):
        least_important = np.argmin(feature_importances)
        removed_columns = removed_columns.append(X_train_opt.pop(X_train_opt.columns[least_important]))
        model.fit(X_train_opt, y_train)
        feature_importances = model.feature_importances_
        accuracies = cross_val_score(estimator = model,
                                    X = X_train_opt,
                                    y = y_train, cv = 5,
                                    scoring = 'r2')
        r2s_opt = np.append(r2s_opt, accuracies.mean())
        models_features = np.append(models_features, ", ".join(list(X_train_opt)))
        
    feature_selection = pd.DataFrame({'Features':models_features,'r2_Score':r2s_opt})
    return feature_selection