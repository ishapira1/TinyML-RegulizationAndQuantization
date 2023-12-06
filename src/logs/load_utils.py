
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
# Load the CSV file into a pandas DataFrame
from datetime import datetime

import numpy as np

# Set the aesthetics for the seaborn plots
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_palette("Set2")
plt.rcParams.update({"xtick.labelsize": 17, "ytick.labelsize": 17,
"axes.titlesize": 18, "axes.titleweight": 'bold', "axes.labelsize": 19, "axes.labelweight": 'bold'})
pd.set_option('display.max_columns', 50)


def load_dataframe(path="../results/all_results.csv", columns=None):
    # Load the DataFrame
    df = pd.read_csv(path)
    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d_%H-%M-%S')

    # Select only the specified columns if the 'columns' parameter is provided
    if columns is not None:
        df = df[columns]

    # Remove rows where 'total weight' is NaN
    df = df.dropna(subset=['total_weights'])
    df = df[(df.regularization != 'dropout') & (df.regularization != 'l_infinty')]
    return df

def create_pivot_table(df, model_name, dataset_name, regularization, metric='test_accuracy'):
    # Filter the DataFrame based on the input parameters
    filtered_df = df[(df['model_name'] == model_name) & 
                     (df['dataset_name'] == dataset_name) & 
                     ((df['regularization'] == regularization) | (df['regularization'] == 'none'))]

    # List of bit widths to consider
    bit_widths = [32, 16, 8, 4, 2]
    # Create a new DataFrame for the pivot table
    pivot_data = []

    for bit in bit_widths:
        if bit == 32:
            # For bit width 32, use the 'test_loss' column
            col_name = metric
        else:
            # For other bit widths, use the 'bit_{i}_{metric}' format
            col_name = f'bit_{bit}_{metric}'

        # Check if the column exists in the DataFrame
        if col_name in filtered_df.columns:
            # Extract the required values and add to the pivot data
            for index, row in filtered_df.iterrows():
                pivot_data.append({'Bit Width': f'bit_{bit}', 
                                   'Regularization Param': row['regularization_param'], 
                                   metric: row[col_name]})

    # Convert the list of data into a DataFrame
    pivot_df = pd.DataFrame(pivot_data)


    # Sort the DataFrame by 'Bit Width' in descending order
    pivot_df['Bit Width'] = pd.Categorical(pivot_df['Bit Width'], 
                                           categories=[f'bit_{b}' for b in bit_widths], 
                                           ordered=True)
    pivot_df = pivot_df.sort_values('Bit Width', ascending=False)

    # Create the pivot table
    pivot_table = pivot_df.pivot(index='Bit Width', columns='Regularization Param', values=metric)

    return pivot_table

def my_eval(x):
    try:
        return eval(x)
    except:
        print(x)
        return x

