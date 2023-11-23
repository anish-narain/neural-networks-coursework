import io
import itertools

import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
from sklearn.model_selection import train_test_split

class Regressor(nn.Module):

    def __init__(self, x, scaler="minmax", learning_rate=0.01, batch_size=512, loss="mse", shuffle_flag=True, num_hidden_layers=3, nb_epoch=1000, neurons=None, activations=None):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        super(Regressor, self).__init__()

        # Determine input and output layer sizes
        self.label_binarizer = LabelBinarizer()
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.shuffle_flag = shuffle_flag
        self.learning_rate = learning_rate
        self.num_hidden_layers = num_hidden_layers

        neurons = [10, 10] if neurons is None else [neurons] * self.num_hidden_layers
        activations = ["relu", "relu"] if activations is None else [activations] * self.num_hidden_layers

        # Construct network structure
        neurons = [self.input_size, *neurons, self.output_size]
        activations.append("identity")  # output layer

        layers = []
        for i in range(len(neurons) - 1):
            # Linear layer
            layers.append(nn.Linear(neurons[i], neurons[i + 1]))
            # Activation
            if activations[i] == "relu":
                layers.append(nn.ReLU())
            elif activations[i] == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activations[i] == "tanh":
                layers.append(nn.Tanh())

        self.layers = nn.Sequential(*layers)

        # type of scaler
        scalers = {"minmax": MinMaxScaler(), "maxabs": MaxAbsScaler(), "robust": RobustScaler(), "standard": StandardScaler()}
        self.scaler = scalers[scaler]

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # type of loss function
        loss_function = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}
        self.loss_function = loss_function[loss]
        
        return

    def forward(self, x):
        return self.layers(x)

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        # Separate numerical and categorical columns
        numerical_cols = x.select_dtypes(include=['number'])
        categorical_cols = x.select_dtypes(exclude=['number'])

    
        # Fill NaN values in numerical columns
        x[numerical_cols.columns] = x[numerical_cols.columns].apply(lambda col: col.fillna(col.mean())) 

        if training:
            # Only fit the LabelBinarizer on training data
            if not categorical_cols.empty:
                categorical_cols = self.label_binarizer.fit_transform(categorical_cols)
            if isinstance(y, pd.DataFrame):
                y = y.apply(lambda col: self.scaler.fit_transform(col.values.reshape(-1, 1)).flatten())
        else:
            if not categorical_cols.empty:
                categorical_cols = self.label_binarizer.transform(categorical_cols)
            if isinstance(y, pd.DataFrame):
                y.apply(lambda col: self.scaler.transform(col.values.reshape(-1, 1)).flatten())

        # Combine numerical and categorical columns back
        x = pd.concat([numerical_cols, pd.DataFrame(categorical_cols)], axis=1)

        return x, (y if isinstance(y, pd.DataFrame) else None)


    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """
        # Convert preprocessed data to PyTorch tensors
        X, Y = self._preprocessor(x, y=y, training=True)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

        # Create a DataLoader for batch processing
        dataset = TensorDataset(X_tensor, Y_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle_flag)

        # Training loop
        for i in range(self.nb_epoch):
            for batch_x, batch_y in data_loader:

                # Forward pass
                outputs = self.forward(batch_x)
                loss = self.loss_function(outputs, batch_y)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()  
                self.optimizer.step()  
        return self

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """
        X, _ = self._preprocessor(x, training=False)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        # inherited method that sets the model to evaluate mode
        self.eval()

        with torch.no_grad():  
            predictions = self(X_tensor)

        # Convert the predictions back to a NumPy array
        predictions_np = predictions.numpy()

        # Invert y-value scaling form preprocessor
        predictions_np = self.scaler.inverse_transform(predictions_np)

        return predictions_np


    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """
        # Preprocess the input and target data
        _, Y_true = self._preprocessor(x, y=y, training=False)
        Y_true = self.scaler.inverse_transform(Y_true)  # Scale back up

        # Use the predict method to make predictions
        Y_predicted = self.predict(x)


        # Convert Y_true to a NumPy array if it's not already
        Y_true = Y_true if isinstance(Y_true, np.ndarray) else Y_true.values

        if np.isnan(np.min(Y_true)):
            print("y true")
        elif np.isnan(np.min(Y_predicted)):
            print("y predicted")
       
       # Calculating different evaluation metrics

        mse = mean_squared_error(Y_true, Y_predicted)
        # mae = mean_absolute_error(Y_true, Y_predicted)
        # rmse = mean_squared_error(Y_true, Y_predicted, squared=False)
        # r2 = r2_score(Y_true, Y_predicted)
        # explained_variance = explained_variance_score(Y_true, Y_predicted)
        #
        # # MAPE - Mean Absolute Percentage Error
        # mape = np.mean(np.abs((Y_true - Y_predicted) / Y_true)) * 100
        #
        # metrics = {
        #     "Mean Absolute Error (MAE)": round(mae, 4),
        #     "Mean Squared Error (MSE)": round(mse, 4),
        #     "Root Mean Squared Error (RMSE)": round(rmse, 4),
        #     "R^2 Score": round(r2, 4),
        #     "Mean Absolute Percentage Error (MAPE)": round(mape, 4),
        #     "Explained Variance Score": round(explained_variance, 4),
        # }
        # #print(metrics)
        return mse



def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    #print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    #print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(train, val):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """
    # Define ranges for hyperparameters
    num_hidden_layers = [3, 5]
    num_neurons = [64, 128]
    loss_funcs = ["mse", "mae"]
    activation_functions = ["relu", "sigmoid", "tanh"]
    scalers = ["minmax", "maxabs", "robust", "standard"]
    batch_sizes = [512, 1024]
    epochs_range = [50, 100, 200]
    learning_rates = [0.001, 0.01]

    parameters = itertools.product(*[num_hidden_layers, num_neurons, loss_funcs, activation_functions, scalers, batch_sizes, epochs_range, learning_rates])
    best_score = float('inf')
    best_params = {}

    x_train = train.drop('median_house_value', axis=1)
    y_train = train[['median_house_value']]

    x_val = val.drop('median_house_value', axis=1)
    y_val = val[['median_house_value']]


    file = open("values.csv", 'a')
    csv_writer= csv.writer(file)

    i = 0

    csv_writer.writerow(["hidden layer", "neurons", "loss", "activations", "scaler", "batch_size", "nb_epochs", "learning_rate", "score"])
    for hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate in parameters:
        model = Regressor(x=x_train, scaler=scaler, learning_rate=learning_rate, batch_size=batch_size, loss=loss, nb_epoch=nb_epochs, num_hidden_layers=hidden_layers, neurons=neurons, activations=activations)
        model.fit(x_train, y_train)
        score = model.score(x_val, y_val)
        csv_writer.writerow([hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate, score])
        print(i, [hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate, score])
        i += 1
        if score < best_score:
            best_score = score
            best_params = {"hidden_layer": hidden_layers, "neurons": neurons, "loss": loss,"activations": activations, "scaler":scaler, "batch_size" :batch_size, "nb_epochs": nb_epochs, "learning_rate": learning_rate}
    file.close()

    print("best parameters")
    print(best_params)
    return best_params

def plot_features_for_report(df):
    # Numeric features to plot
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()
    # Categorical features to plot
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Font settings
    plt.rcParams['font.family'] = 'Cambria'
    
    # Function to format feature names (capitalizes and replaces underscores with spaces)
    def format_feature_name(name):
        return ' '.join(word.capitalize() for word in name.replace('_', ' ').split())
    
    # Plotting numeric features with histograms
    for feature in numeric_features:
        plt.figure(figsize=(10, 4))
        df[feature].hist(bins=20, color='purple', edgecolor='black', rwidth=0.8)  # Adjust bar width
        plt.yscale('log')
        feature_title = format_feature_name(feature) + ' Frequency'  # Format feature name for the title
        plt.title(feature_title)
        plt.ylabel('Frequency')
        plt.xlabel(format_feature_name(feature))  # Format feature name for the x-axis label
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid lines and adjust transparency
        plt.show()
    
    # Plotting categorical features with bar plots
    for feature in categorical_features:
        plt.figure(figsize=(10, 4))
        series = df[feature].value_counts()
        series.plot(kind='bar', color='purple', edgecolor='black', width=0.8)  # Adjust bar width and color
        feature_title = format_feature_name(feature) + ' Count'  # Format feature name for the title
        plt.title(feature_title)
        plt.ylabel('Count')
        plt.xlabel(format_feature_name(feature))  # Format feature name for the x-axis label
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Add grid lines and adjust transparency
        plt.show()

def calculate_missing_percentage_for_report(df):
    """
    This function takes a pandas DataFrame as input and prints out the percentage of missing values
    for each column in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame for which to calculate missing value percentages.

    Returns:
    None
    """
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().mean() * 100
    
    # Print the percentage of missing values for each column
    print("Percentage of missing values for each column:")
    for column, percentage in missing_percentage.items():
        print(f"{column}: {percentage:.2f}%")
    
    return None  # Explicitly return None for clarity
        
def main():
    # Load data
    df = pd.read_csv('housing.csv')

    # Assuming the target variable is 'median_house_value'
    X = df.drop('median_house_value', axis=1)
    y = df[['median_house_value']]

    # Initialize the Regressor with the data
    regressor = Regressor(df)

    # Preprocess and split the data
    processed, _ = regressor._preprocessor(df, training=True)


    #hyperparameter tuning
    train, test = train_test_split(processed, test_size = 0.2, random_state = 12)

    print("hyperparameter tuning")
    RegressorHyperParameterSearch(train, test)

def test_preprocessor():
    df = pd.read_csv('housing.csv')
    regressor = Regressor(df)
    preprocessed_X, _ = regressor._preprocessor(df, training=True)
    print(preprocessed_X.head())  # Display the first few rows of the preprocessed data
    

if __name__ == "__main__":
    main()
    #test_preprocessor()
    df = pd.read_csv('housing.csv')
    #plot_features_for_report(df)
    #calculate_missing_percentage_for_report(df)


