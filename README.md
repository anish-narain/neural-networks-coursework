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

This function will iterate over a predefined set of hyperparameters to find the combination that produces the best results on the validation dataset. This parameters were [hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate].


## Visualization and Analysis

Two additional functions are provided to help with data analysis and visulaisation of the data:

**plot_features_for_report(df):** This function generates histograms and bar plots for the features in your dataset.
**calculate_missing_percentage_for_report(df):** This function calculates and prints the percentage of missing values for each column in your dataset.
Uncomment the respective function calls in the **if __name__ == "__main__":** block to use them.

