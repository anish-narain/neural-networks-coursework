import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

class Regressor(nn.Module):

    def __init__(self, x, scaler, optimizer, batch_size, loss, shuffle_flag=True, nb_epoch = 1000):
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
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.shuffle_flag = shuffle_flag
        self.label_binarizer = LabelBinarizer()

        scalers = {"minmax": MinMaxScaler(), "maxabs": MaxAbsScaler(), "robust": RobustScaler(), "standard": StandardScaler()}
        self.scaler = scalers[scaler]

        optimizers = {"adam": torch.optim.Adam(self.parameters(), lr=0.001), "sgd": torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)}
        self.optimizer = optimizers[optimizer]

        loss_function = {"mse": nn.MSELoss(), "mae": nn.L1Loss()}
        self.loss_function = loss_function[loss]
        
        return

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
        x = x.apply(lambda column: column.fillna(column.mean()))

        # Normalize numerical variables
        numerical_features = x.select_dtypes(include=[np.number]).columns
        if training:
            # Fit and transform for training data
            x['ocean_proximity'] = self.label_binarizer.fit_transform(x['ocean_proximity'])
            x[numerical_features] = self.scaler.fit_transform(x[numerical_features])
            if y: y = self.scaler.fit_transform(y.values.reshape(-1, 1))
        else:
            # Transform for test/validation data
            x['ocean_proximity'] = self.label_binarizer.transform(x['ocean_proximity'])
            x[numerical_features] = self.scaler.transform(x[numerical_features])
            if y: y = self.scaler.transform(y.values.reshape(-1, 1))

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
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

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
       
       # Calculating different evaluation metrics
        mae = mean_absolute_error(Y_true, Y_predicted)
        mse = mean_squared_error(Y_true, Y_predicted)
        rmse = mean_squared_error(Y_true, Y_predicted, squared=False)
        r2 = r2_score(Y_true, Y_predicted)
        explained_variance = explained_variance_score(Y_true, Y_predicted)

        # MAPE - Mean Absolute Percentage Error
        mape = np.mean(np.abs((Y_true - Y_predicted) / Y_true)) * 100

        metrics = {
            "Mean Absolute Error (MAE)": round(mae, 4),
            "Mean Squared Error (MSE)": round(mse, 4),
            "Root Mean Squared Error (RMSE)": round(rmse, 4),
            "R^2 Score": round(r2, 4),
            "Mean Absolute Percentage Error (MAPE)": round(mape, 4),
            "Explained Variance Score": round(explained_variance, 4),
        }
        print(metrics)
        return mse



def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    # Identifying ideal number of neurons

    # Identifying ideal activation function

    # Identifying ideal optimizer

    # Identifying ideal scaler type

    # Identifying ideal batch size


    return  # Return the chosen hyper parameters



""""
def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

"""

def main():
    # Load data
    df = pd.read_csv('housing.csv')
    # Assuming the target variable is 'median_house_value'
    X = df.drop('median_house_value', axis=1)
    y = df[['median_house_value']]

    # Initialize the Regressor with the data
    regressor = Regressor(X)

    # Preprocess and split the data
    X_processed, y_processed = regressor._preprocessor(X, y, training=True)
    print("Processed X:", X_processed)
    print("Processed y:", y_processed)

def test_preprocessor():
    df = pd.read_csv('housing.csv')
    regressor = Regressor(df)
    preprocessed_X, _ = regressor._preprocessor(df, training=True)
    preprocessed_X.head()  # Display the first few rows of the preprocessed data
    

if __name__ == "__main__":
    #main()
    test_preprocessor()


