from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


def evaluate_model(y_test, y_pred):
    exp_var_score = explained_variance_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    # print('explained variance = {}'.format(explained_variance_score))
    # print('mse = {}'.format(mean_squared_error))
    # print('r2 = {}'.format(r_squared))
    return exp_var_score, mse, r_squared