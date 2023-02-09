from reader import load_input_data
from preproccessing import describtive_statistics, convert_and_add_timecol, agg_data, data_split
from visualization import (feature_hist_dist_plot, corr_matrix_plot, radiation_by_hour_join_plot, 
                           radiation_by_timeseries, feature_important_bar_chart, visualize_prediction_vs_actual)
from model import prediction_model, feature_important
from model_evaluation import evaluate_model

if __name__ == "__main__":
    '''
    Main Flow to run model:
    I. Load input data
    II. Preprocess data
    III. EDA
    IV. Modeling  
    V. Model Evaluation
    
    '''
    # 1. Load input data
    data_path = "./dataset/SolarPrediction.csv"
    dataset = load_input_data(data_path)
    #print(dataset.head())

    # 2. Descriptive Statistic
    print("Descriptive statistic", describtive_statistics(dataset))

    # 3. Convert and add time column
    df = convert_and_add_timecol(dataset)
    #print(df.columns)

    # 4. Data visualization (EDA)

    # Visualize Ratation timeseries data
    col = "Radiation"
    radiation_by_timeseries(df, col)

    
    # Aggregate data by specific time
    grouped_d, grouped_m, grouped_d, grouped_h = agg_data(df)
    #print(grouped_h)

    # Visualize Histogram distribution of Mean radiation, Temperature, Pressure, Humulity
    feature_hist_dist_plot(grouped_d, grouped_m, grouped_d)
    
    # Visualize join plot of Radiation by hour
    radiation_by_hour_join_plot(grouped_h, "TimeOfDay(h)", "Radiation")

    # Visualize correlationmatrix
    corr_matrix_plot(df)

    # 5. Modelling
    X = df[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed', 'DayOfYear', 'TimeOfDay(s)']]
    y = df['Radiation']
    X_train, X_test, y_train, y_test =data_split(X, y)
    model = prediction_model(X_train, y_train, X_test)
    #print(y_pred)
    feature_selection = feature_important(model,X_train, y_train)
    print(feature_selection.head())

    # Visualize feature important
    # data_feauture = feature_selection.head()
    features = "Features"
    score = "r2_Score"
    feature_important_bar_chart(feature_selection.head(10), features, score)

    # 6. Model Evaluation result
    # Print evaluation report    
    X_train_best = X_train[['Temperature', 'DayOfYear', 'TimeOfDay(s)']] # Just select 3 key features base on the feature important result above
    X_test_best = X_test[['Temperature', 'DayOfYear', 'TimeOfDay(s)']] # Just select 3 key features base on the feature important result above
    model_best = prediction_model(X_train_best, y_train, X_test_best)
    y_pred = model_best.predict(X_test_best)
    exp_var_score, mse, r_squared = evaluate_model(y_test, y_pred)
    print('explained variance = {}'.format(exp_var_score))
    print('mse = {}'.format(mse))
    print('r2 = {}'.format(r_squared))

    # Visualize prediction vs actual chart
    df['y_pred'] = model_best.predict(df[['Temperature', 'DayOfYear', 'TimeOfDay(s)']])
    print(df.head())

    actual = "Radiation"
    predict = "y_pred"
    visualize_prediction_vs_actual(df, actual, predict)

