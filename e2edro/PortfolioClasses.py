# PortfolioClasses Module
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

####################################################################################################
# SlidingWindow torch Dataset to index data to use a sliding window
####################################################################################################
class SlidingWindow(Dataset):
    """Sliding window dataset constructor
    """
    def __init__(self, X, Y, n_obs, perf_period):
        """Construct a sliding (i.e., rolling) window dataset from a complete timeseries dataset

        Inputs
        X: pandas dataframe with the complete feature dataset
        Y: pandas dataframe with the complete asset return dataset
        n_obs: Number of scenarios in the window
        perf_period: Number of scenarios in the 'performance window' used to evaluate out-of-sample
        performance. The 'performance window' is also a sliding window

        Output
        Dataset where each element is the tuple (x, y, y_perf)
        x: Feature window (dim: [n_obs+1] x n_x)
        y: Realizations window (dim: n_obs x n_y)
        y_perf: Window of forward-looking (i.e., future) realizations (dim: perf_period x n_y)

        Note: For each feature window 'x', the last scenario x_t is reserved for prediction and
        optimization. Therefore, no pair in 'y' is required (it is assumed the pair y_T is not yet
        observable)
        """
        # Handle both pandas DataFrames and numpy arrays
        if hasattr(X, 'values'):
            X_data = X.values
        elif hasattr(X, 'to_numpy'):
            X_data = X.to_numpy()
        else:
            X_data = X
            
        if hasattr(Y, 'values'):
            Y_data = Y.values
        elif hasattr(Y, 'to_numpy'):
            Y_data = Y.to_numpy()
        else:
            Y_data = Y
        
        # Ensure we have numpy arrays
        if not isinstance(X_data, np.ndarray):
            X_data = np.array(X_data)
        if not isinstance(Y_data, np.ndarray):
            Y_data = np.array(Y_data)
        
        self.X = Variable(torch.tensor(X_data, dtype=torch.double))
        self.Y = Variable(torch.tensor(Y_data, dtype=torch.double))
        self.n_obs = n_obs
        self.perf_period = perf_period

    def __getitem__(self, index):
        x = self.X[index:index+self.n_obs+1]
        y = self.Y[index:index+self.n_obs]
        y_perf = self.Y[index+self.n_obs : index+self.n_obs+self.perf_period+1]
        return (x, y, y_perf)

    def __len__(self):
        return len(self.X) - self.n_obs - self.perf_period

####################################################################################################
# Backtest object to store out-of-sample results
####################################################################################################
class backtest:
    """backtest object
    """
    def __init__(self, len_test, n_y, dates):
        """Portfolio object. Stores the NN out-of-sample results

        Inputs
        len_test: Number of scenarios in the out-of-sample evaluation period
        n_y: Number of assets in the portfolio
        dates: DatetimeIndex 

        Output
        Backtest object with fields:
        weights: Asset weights per period (dim: len_test x n_y)
        rets: Realized portfolio returns (dim: len_test x 1)
        tri: Total return index (i.e., absolute cumulative return) (dim: len_test x 1)
        mean: Average return over the out-of-sample evaluation period (dim: scalar)
        vol: Volatility (i.e., standard deviation of the returns) (dim: scalar)
        sharpe: pseudo-Sharpe ratio defined as 'mean / vol' (dim: scalar)
        """
        self.weights = np.zeros((len_test, n_y))
        self.rets = np.zeros(len_test)
        self.dates = dates[-len_test:]

    def stats(self):
        tri = np.cumprod(self.rets + 1)
        self.mean = (tri[-1])**(1/len(tri)) - 1
        self.vol = np.std(self.rets)
        self.sharpe = self.mean / self.vol
        
        # Store results as numpy arrays to avoid pandas corruption
        self.rets_array = np.column_stack([self.dates, self.rets, tri])
        self.rets = self.rets_array  # Store as numpy array instead of pandas DataFrame

####################################################################################################
# InSample object to store in-sample results
####################################################################################################
class InSample:
    """InSample object
    """
    def __init__(self):
        """Portfolio object. Stores the NN in-sample results

        Output
        InSample object with fields:
        loss: Empty list to hold the training loss after each forward pass
        gamma: Empty list to hold the gamma value after each backward pass
        delta: Empty list to hold the delta value after each backward pass
        val_loss (optional): Empty list to hold the valildation loss after each forward pass
        """
        self.loss = []
        self.gamma = []
        self.delta = []
        self.val_loss = []

    def df(self):
        """Return a pandas dataframe object by merging the self.lists
        """
        # Convert numpy arrays and tensors to Python native types for pandas compatibility
        loss_clean = [float(l.item()) if hasattr(l, 'item') else float(l) for l in self.loss]
        gamma_clean = [float(g.item()) if hasattr(g, 'item') else float(g) for g in self.gamma]
        delta_clean = [float(d.item()) if hasattr(d, 'item') else float(d) for d in self.delta]
        val_loss_clean = [float(vl.item()) if hasattr(vl, 'item') else float(vl) for vl in self.val_loss]
        
        if not self.delta and not self.val_loss:
            return pd.DataFrame(list(zip(loss_clean, gamma_clean)), columns=['loss', 'gamma'])
        elif not self.delta:
            return pd.DataFrame(list(zip(loss_clean, val_loss_clean, gamma_clean)), 
                            columns=['loss', 'val_loss', 'gamma'])
        elif not self.val_loss:
            return pd.DataFrame(list(zip(loss_clean, gamma_clean, delta_clean)), 
                            columns=['loss', 'gamma', 'delta'])
        else:
            return pd.DataFrame(list(zip(loss_clean, val_loss_clean, gamma_clean, delta_clean)), 
                            columns=['loss', 'val_loss', 'gamma', 'delta'])


####################################################################################################
# Backtest object to store out-of-sample results
####################################################################################################
class CrossVal:
    """Portfolio object
    """
    def __init__(self):
        """CrossVal object. Stores the NN in-sample cross validation results

        Output
        CrossVal object with fields:
        lr: Empty list to hold the learning rate of this run
        epochs: Empty list to hold the number of epochs in this run
        train_loss: Empty list to hold the average training loss of all folds
        val_loss: Empty list to hold the average validation loss of all folds
        """
        self.lr = []
        self.epochs = []
        self.val_loss = []

    def df(self):
        """Return a pandas dataframe object by merging the self.lists
        """
        # Convert numpy arrays and tensors to Python native types for pandas compatibility
        lr_clean = [float(lr.item()) if hasattr(lr, 'item') else float(lr) for lr in self.lr]
        epochs_clean = [int(epoch.item()) if hasattr(epoch, 'item') else int(epoch) for epoch in self.epochs]
        val_loss_clean = [float(vl.item()) if hasattr(vl, 'item') else float(vl) for vl in self.val_loss]
        
        return pd.DataFrame(list(zip(lr_clean, epochs_clean, val_loss_clean)), 
                            columns=['lr', 'epochs', 'val_loss'])

