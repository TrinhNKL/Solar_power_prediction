import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd 

def load_input_data(data_path):
    """
        Descrintion: Loading input csv data
                
        Parameters:
        data_patg: Filepath Where inputs are Stored.
        display: True/False Boolean to display.
        Returns:
        - dataframe
        - datafrae shape
    """
    data = pd.read_csv(data_path)  
    data = data.sort_values(['UNIXTime'], ascending = [True])
    return data

print('Finish load data')