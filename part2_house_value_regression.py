import itertools
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from part1_nn_lib import Preprocessor


class Regressor(nn.Module):

    def __init__(self, x, scaler="minmax", learning_rate=0.01, batch_size=512, loss="mse", shuffle_flag=True,
                 num_hidden_layers=3, nb_epoch=1000, neurons=None, activations=None):
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

        scalers = {"minmax": MinMaxScaler(), "maxabs": MaxAbsScaler(), "robust": RobustScaler(),
                   "standard": StandardScaler()}
        self.scaler = scalers[scaler]

        X, _ = self._preprocessor(x, training=True)
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
        activations.append("identity")

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # type of loss function
        loss_function = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}
        self.loss_function = loss_function[loss]

        return

    def forward(self, x):
        return self.layers(x)

    def _preprocessor(self, x, y=None, training=False):
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
        categorical_feature = "ocean_proximity"
        numerical_features = x.select_dtypes(include=[np.number]).columns
        if training:
            self.means = x.mean(numeric_only=True)

        x = x.fillna(self.means)
        if y is not None:
            y = y.fillna(y.mean(numeric_only=True))

        if training:
            self.label_binarizer.fit(x[categorical_feature])
            x[numerical_features] = self.scaler.fit_transform(x[numerical_features])
        else:
            x[numerical_features] = self.scaler.transform(x[numerical_features])

        binarized_x = x.join(pd.DataFrame(self.label_binarizer.transform(x[categorical_feature]), columns=self.label_binarizer.classes_,
                         index=x.index)).drop(categorical_feature, axis=1)
        x_numpy = binarized_x.to_numpy()
        if training:
            self.preprocessor = Preprocessor(x_numpy)

        return x_numpy, (y.to_numpy() if y is not None else None)

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
        x_numpy, y = self._preprocessor(x, y=y, training=True)
        X = torch.tensor(self.preprocessor.apply(x_numpy).astype(np.float32))
        Y = torch.tensor(y.astype(np.float32)) if y is not None else None

        # Create a DataLoader for batch processing
        dataset = TensorDataset(X, Y)
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
        x_numpy, _ = self._preprocessor(x, training=False)
        X = torch.tensor(self.preprocessor.apply(x_numpy).astype(np.float32))

        # inherited method that sets the model to evaluate mode
        self.eval()

        with torch.no_grad():
            predictions = self(X)

        # Convert the predictions back to a NumPy array
        predictions_np = predictions.numpy()

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

        # Use the predict method to make predictions
        Y_predicted = self.predict(x)

        rmse = mean_squared_error(Y_true, Y_predicted, squared=False)
        return rmse


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    # print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    # print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train, x_val, y_train, y_val):
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """
    # Define ranges for hyperparameters
    num_hidden_layers = [5, 7, 9]
    num_neurons = [64, 128, 256]
    loss_funcs = ["mse"]
    activation_functions = ["relu"]
    scalers = ["robust"]
    batch_sizes = [128, 256, 512]
    epochs_range = [100, 200, 500]
    learning_rates = [0.01, 0.1, 0.25]

    parameters = itertools.product(
        *[num_hidden_layers, num_neurons, loss_funcs, activation_functions, scalers, batch_sizes, epochs_range,
          learning_rates])
    best_score = float('inf')
    best_params = {}
    best_regressor = None

    file = open("values2.csv", 'a')
    csv_writer = csv.writer(file)

    i = 0

    csv_writer.writerow(
        ["counter", "hidden layer", "neurons", "loss", "activations", "scaler", "batch_size", "nb_epochs",
         "learning_rate",
         "score"])
    for hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate in parameters:
        model = Regressor(x=x_train, scaler=scaler, learning_rate=learning_rate, batch_size=batch_size, loss=loss,
                          nb_epoch=nb_epochs, num_hidden_layers=hidden_layers, neurons=neurons, activations=activations)
        model.fit(x_train, y_train)
        score = model.score(x_val, y_val)

        csv_writer.writerow(
            [i, hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate, score])
        print(i, [hidden_layers, neurons, loss, activations, scaler, batch_size, nb_epochs, learning_rate, score])
        i += 1

        if score < best_score:
            best_score = score
            best_params = {"hidden_layer": hidden_layers, "neurons": neurons, "loss": loss, "activations": activations,
                           "scaler": scaler, "batch_size": batch_size, "nb_epochs": nb_epochs,
                           "learning_rate": learning_rate}
            best_regressor = model

    save_regressor(best_regressor)
    csv_writer.writerow(["best parameters"])
    csv_writer.writerow(best_params.values())
    file.close()
    return best_params


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


def ScoreBestModel(x_test, y_test):
    """
    Evaluates the performance of the best model by loading it, scoring it on the provided test set,
    and printing the obtained score. Additionally, the score is appended to the 'values2.csv' file.

    Arguments:
        x_test (DataFrame): Features of the test set.
        y_test (DataFrame): Target values of the test set.
    """
    best_model = load_regressor()
    score = best_model.score(x_test, y_test)
    print("best model score: ", score)
    file = open("values2.csv", 'a')
    csv_writer = csv.writer(file)
    csv_writer.writerow([score])


def main():
    # Load data
    df = pd.read_csv('housing.csv')

    # Splitting data into testing and training datasets
    x_train, x_temp, y_train, y_temp = train_test_split(df, df[["median_house_value"]], test_size=0.2, random_state=12)
    x_train = x_train.drop('median_house_value', axis=1)

    # Further splitting data into validation and testing datasets
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=12)

    x_val = x_val.drop('median_house_value', axis=1)
    x_test = x_test.drop('median_house_value', axis=1)

    # Passing training and validation datasets to be used for hyper-parameter tuning
    RegressorHyperParameterSearch(x_train, x_val, y_train, y_val)

    # Testing the model that produced the lowest root mean squared error using the separate testing dataset
    ScoreBestModel(x_test, y_test)


def test_preprocessor():
    df = pd.read_csv('housing.csv')
    regressor = Regressor(df)
    preprocessed_X, _ = regressor._preprocessor(df, training=True)
    print(preprocessed_X.head())  # Display the first few rows of the preprocessed data


if __name__ == "__main__":
    main()
