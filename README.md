# Machine Learning Neural Network Coursework 2

# House Value Regression Model

This repository contains a Python program that trains a neural network to predict house values based on various input features. The model is built using the PyTorch framework and includes data preprocessing, model training, and evaluation.

## Prerequisites

Before running the code, ensure you have the following packages installed:

- Python 3.8 or newer
- PyTorch
- pandas
- NumPy
- Matplotlib
- scikit-learn

You can install these packages using the following command:

```
pip install torch pandas numpy matplotlib scikit-learn
```

## Hyperparameter Tuning

The script includes a function for hyperparameter tuning:

```
RegressorHyperParameterSearch(train, test)
```

This function will iterate over a predefined set of hyperparameters to find the combination that produces the best results on the validation dataset. This parameters were [hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate]. The function saves the results in a CSV file, loads the best model to a Pickle file, and returns the optimised hyperparameters.

The script also includes a function to test the model loaded into **part2_model.pickle**:

```
ScoreBestModel(x_test, y_test):
```
This function loads the model, calculates its score on a separate test set, prints the result, and appends the score to 'values2.csv'. We used this function to analyse our best model in our report.


## Visualization and Analysis

Three additional functions are provided to help with data analysis and visualisation of the data, and can be found in the **plot.py** file:

**plot_features_for_report(df):** This function generates histograms and bar plots for the features in the dataset.

**calculate_missing_percentage_for_report(df):** This function calculates and prints the percentage of missing values for each column in the dataset.

**plot_results_of_hyperparameter_search(values):** This functions generates bar plots and line graphs to allow us to visualise the impact of varying hyperparameters on the MSE loss of the predictions of the neural network. This allowed us to further fine tune our hyperparameters as detailed in our report.

## Running the Code

To execute this code successfully, ensure to run the `main` function by executing the script. The main function loads the data, splits it into training, validation, and testing datasets, conducts hyper-parameter tuning, and evaluates the best model's performance on a separate testing dataset.
