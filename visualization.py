import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go



def feature_hist_dist_plot(grouped_d, grouped_m, grouped_h):
    f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, sharex='col', sharey='row', figsize=(14,12))
    ax3.set_ylim(45,60)
    ax5.set_ylim(30.36,30.46)
    ax7.set_ylim(60,85)

    ax1.set_title('Mean Radiation by Hour')
    pal = sns.color_palette("YlOrRd_r", len(grouped_h))
    rank = grouped_h['Radiation'].argsort().argsort() 
    g = sns.barplot(x="TimeOfDay(h)", y='Radiation', data=grouped_h, palette=np.array(pal[::-1])[rank], ax=ax1)
    ax1.set_xlabel('')

    ax2.set_title('Mean Radiation by Month')
    pal = sns.color_palette("YlOrRd_r", len(grouped_m))
    rank = grouped_m['Radiation'].argsort().argsort() 
    g = sns.barplot(x="MonthOfYear", y='Radiation', data=grouped_m, palette=np.array(pal[::-1])[rank], ax=ax2)
    ax2.set_xlabel('')

    ax3.set_title('Mean Temperature by Hour')
    pal = sns.color_palette("YlOrRd_r", len(grouped_h))
    rank = grouped_h['Temperature'].argsort().argsort() 
    g = sns.barplot(x="TimeOfDay(h)", y='Temperature', data=grouped_h, palette=np.array(pal[::-1])[rank], ax=ax3)
    ax3.set_xlabel('')

    ax4.set_title('Mean Temperature by Month')
    pal = sns.color_palette("YlOrRd_r", len(grouped_m))
    rank = grouped_m['Temperature'].argsort().argsort() 
    g = sns.barplot(x="MonthOfYear", y='Temperature', data=grouped_m, palette=np.array(pal[::-1])[rank], ax=ax4)
    ax4.set_xlabel('')

    ax5.set_title('Mean Pressure by Hour')
    pal = sns.color_palette("YlOrRd_r", len(grouped_h))
    rank = grouped_h['Pressure'].argsort().argsort() 
    g = sns.barplot(x="TimeOfDay(h)", y='Pressure', data=grouped_h, palette=np.array(pal[::-1])[rank], ax=ax5)
    ax5.set_xlabel('')

    ax6.set_title('Mean Pressure by Month')
    pal = sns.color_palette("YlOrRd_r", len(grouped_m))
    rank = grouped_m['Pressure'].argsort().argsort() 
    g = sns.barplot(x="MonthOfYear", y='Pressure', data=grouped_m, palette=np.array(pal[::-1])[rank], ax=ax6)
    ax6.set_xlabel('')

    ax7.set_title('Mean Humidity by Hour')
    pal = sns.color_palette("YlOrRd_r", len(grouped_h))
    rank = grouped_h['Humidity'].argsort().argsort() 
    g = sns.barplot(x="TimeOfDay(h)", y='Humidity', data=grouped_h, palette=np.array(pal[::-1])[rank], ax=ax7)

    ax8.set_title('Mean Humidity by Month')
    pal = sns.color_palette("YlOrRd_r", len(grouped_m))
    rank = grouped_m['Humidity'].argsort().argsort() 
    g = sns.barplot(x="MonthOfYear", y='Humidity', data=grouped_m, palette=np.array(pal[::-1])[rank], ax=ax8)

    plt.show()

def corr_matrix_plot(data):
    corrmat = data.drop(['TimeOfDay(h)', 'TimeOfDay(m)', 'TimeOfDay(s)', 'UNIXTime', 'MonthOfYear', 'WeekOfYear'], inplace=False, axis=1)
    corrmat = corrmat.corr()
    f, ax = plt.subplots(figsize=(7,7))
    sns.heatmap(corrmat, vmin=-.8, vmax=.8, square=True, cmap = 'coolwarm')
    plt.show()

def radiation_by_hour_join_plot(data, col1, col2):
    # check the effect of `hour` and `day_of_year` on the `SystemProduction`
    sns.set()
    g = sns.jointplot(data=data , x=col1, y=col2)
    g.set_axis_labels(xlabel=("Hour (24-H Format)"), # set x-label
                  ylabel=("Radiation (W/m2"))  # set y-label
    g.fig.suptitle("Hourly Energy Radiation", y=1.03);  # set the title
    plt.show()

# plot the time series data
def radiation_by_timeseries(data, col):
    # sns.set_context("poster")
    # plt.subplots(figsize=(50, 5)) # set the figure dimensions
    # sns.lineplot(x=data.index, y=data[col].values, color="blue")

    # # set the labels and title
    # plt.xlabel("Date")
    # plt.ylabel("Radiation Generation (W/m2)")
    # plt.title("Time Series Data")
    # plt.show()

    fig = px.line(data, x=data.index, y=col, title='Time Series with Rangeslider')
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()

# Feature Importance chart
def feature_important_bar_chart(data_feature, features, score):
    fig, ax = plt.subplots(figsize = (7,10))
    sns.barplot(y = data_feature[features], x = data_feature[score])
    plt.show()

def visualize_prediction_vs_actual(data, actual, predict):
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data[actual],
                            name="Observation Solar Radiation"))

    fig.add_trace(go.Scatter(x=data.index, y=data[predict],
                            name="Predicted Solar Radiation" ))
    fig.update_layout(
    title="Solar Irradiance Predictions",
    legend_title="Solar Irradiance",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

    fig.show()