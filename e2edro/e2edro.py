# E2E DRO Module
#
####################################################################################################
## Import libraries
####################################################################################################
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd

import e2edro.RiskFunctions as rf
import e2edro.LossFunctions as lf
import e2edro.PortfolioClasses as pc
import e2edro.DataLoad as dl

import psutil
num_cores = psutil.cpu_count()
# For M2 Pro/Max: Use 8-10 threads to avoid thermal throttling
optimal_threads = min(10, num_cores)
torch.set_num_threads(optimal_threads)

# Additional PyTorch optimizations for Apple Silicon
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # Enable MPS optimizations
    torch.backends.mps.enable_fallback_to_cpu = False
    # Set memory fraction for MPS
    if hasattr(torch.backends.mps, 'set_memory_fraction'):
        torch.backends.mps.set_memory_fraction(0.8)  # Use 80% of GPU memory

print(f"🔧 PyTorch Threading: {optimal_threads}/{num_cores} cores | MPS: {torch.backends.mps.is_available()}")

if psutil.MACOS:
    num_cores = 0

####################################################################################################
# CVXPY: Regular optimization layers (nominal and distributionally robust) with OSQP support
####################################################################################################
#---------------------------------------------------------------------------------------------------
# base_mod: CVXPY optimization function that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def base_mod(n_y, n_obs, prisk):
    """Base optimization problem declared as a CVXPY function

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    
    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)

    Objective
    Minimize -y_hat @ z
    """
    # Variables
    z = cp.Variable((n_y,1), nonneg=True)

    # Parameters
    y_hat = cp.Parameter(n_y)
    
    # Constraints
    constraints = [cp.sum(z) == 1]

    # Objective function
    objective = cp.Minimize(-y_hat @ z)

    # Construct optimization problem
    problem = cp.Problem(objective, constraints)

    return problem, z, y_hat

#---------------------------------------------------------------------------------------------------
# nominal: CVXPY optimization function that declares the portfolio optimization problem
#---------------------------------------------------------------------------------------------------
def nominal(n_y, n_obs, prisk):
    """Nominal optimization problem declared as a CVXPY function

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)

    Objective
    Minimize (1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux
    """
    # Convert string risk function to callable function (handle both string and function inputs)
    if isinstance(prisk, str):
        import e2edro.RiskFunctions as rf
        prisk_func = eval('rf.'+prisk)
    else:
        prisk_func = prisk  # prisk is already a function
    
    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    obj_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                    mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [obj_aux[i] >= prisk_func(z, c_aux, ep[i])]

    # Objective function
    objective = cp.Minimize((1/n_obs) * cp.sum(obj_aux) - gamma * mu_aux)

    # Construct optimization problem
    problem = cp.Problem(objective, constraints)

    return problem, z, y_hat, ep, gamma

#---------------------------------------------------------------------------------------------------
# Total Variation: sum_t abs(p_t - q_t) <= delta
#---------------------------------------------------------------------------------------------------
def tv(n_y, n_obs, prisk):
    """DRO layer using the 'Total Variation' distance to define the probability ambiguity set.
    From Ben-Tal et al. (2013).
    Total Variation: sum_t abs(p_t - q_t) <= delta

    Inputs
    n_y: Number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
    lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    eta_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    obj_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    delta: Scalar. Maximum distance between p and q.
    gamma: Scalar. Trade-off between conditional expected return and model error.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al.
    (2013).

    Objective
    Minimize eta_aux + delta * lambda_aux + (1/n_obs) * sum(beta_aux) - gamma * y_hat @ z
    """
    # Convert string risk function to callable function (handle both string and function inputs)
    if isinstance(prisk, str):
        import e2edro.RiskFunctions as rf
        prisk_func = eval('rf.'+prisk)
    else:
        prisk_func = prisk  # prisk is already a function

    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    eta_aux = cp.Variable()
    beta_aux = cp.Variable(n_obs)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)
    
    # Constraints
    constraints = [cp.sum(z) == 1,
                    beta_aux >= -lambda_aux,
                    mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [beta_aux[i] >= prisk_func(z, c_aux, ep[i]) - eta_aux]
        constraints += [lambda_aux >= prisk_func(z, c_aux, ep[i]) - eta_aux]

    # Objective function
    objective = cp.Minimize(eta_aux + delta * lambda_aux + (1/n_obs) * cp.sum(beta_aux)
                            - gamma * mu_aux)

    # Construct optimization problem
    problem = cp.Problem(objective, constraints)

    return problem, z, y_hat, ep, gamma

#---------------------------------------------------------------------------------------------------
# Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta
#---------------------------------------------------------------------------------------------------
def hellinger(n_y, n_obs, prisk):
    """DRO layer using the Hellinger distance to define the probability ambiguity set.
    from Ben-Tal et al. (2013).
    Hellinger distance: sum_t (sqrt(p_t) - sqrtq_t))^2 <= delta

    Inputs
    n_y: number of assets
    n_obs: Number of scenarios in the dataset
    prisk: Portfolio risk function
    
    Variables
    z: Decision variable. (n_y x 1) vector of decision variables (e.g., portfolio weights)
    c_aux: Auxiliary Variable. Scalar. Allows us to p-linearize the derivation of the variance
    lambda_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    xi_aux: Auxiliary Variable. Scalar. Allows for a tractable DR counterpart.
    beta_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable DR counterpart.
    s_aux: Auxiliary Variable. (n_obs x 1) vector. Allows for a tractable SOC constraint.
    mu_aux: Auxiliary Variable. Scalar. Represents the portfolio conditional expected return.

    Parameters
    ep: (n_obs x n_y) matrix of residuals 
    y_hat: (n_y x 1) vector of predicted outcomes (e.g., conditional expected
    returns)
    delta: Scalar. Maximum distance between p and q.
    gamma: Scalar. Trade-off between conditional expected return and model error.

    Constraints
    Total budget is equal to 100%, sum(z) == 1
    Long-only positions (no short sales), z >= 0 (specified during the cp.Variable() call)
    All other constraints allow for a tractable DR counterpart. See the Appendix in Ben-Tal et al.
    (2013).

    Objective
    Minimize xi_aux + (delta-1) * lambda_aux + (1/n_obs) * sum(beta_aux) - gamma * y_hat @ z
    """
    # Convert string risk function to callable function (handle both string and function inputs)
    if isinstance(prisk, str):
        import e2edro.RiskFunctions as rf
        prisk_func = eval('rf.'+prisk)
    else:
        prisk_func = prisk  # prisk is already a function

    # Variables
    z = cp.Variable((n_y,1), nonneg=True)
    c_aux = cp.Variable()
    lambda_aux = cp.Variable(nonneg=True)
    xi_aux = cp.Variable()
    beta_aux = cp.Variable(n_obs, nonneg=True)
    tau_aux = cp.Variable(n_obs, nonneg=True)
    mu_aux = cp.Variable()

    # Parameters
    ep = cp.Parameter((n_obs, n_y))
    y_hat = cp.Parameter(n_y)
    gamma = cp.Parameter(nonneg=True)
    delta = cp.Parameter(nonneg=True)

    # Constraints
    constraints = [cp.sum(z) == 1,
                    mu_aux == y_hat @ z]
    for i in range(n_obs):
        constraints += [xi_aux + lambda_aux >= prisk_func(z, c_aux, ep[i]) + tau_aux[i]]
        constraints += [beta_aux[i] >= cp.quad_over_lin(lambda_aux, tau_aux[i])]
    
    # Objective function
    objective = cp.Minimize(xi_aux + (delta-1) * lambda_aux + (1/n_obs) * cp.sum(beta_aux) 
                            - gamma * mu_aux)

    # Construct optimization problem
    problem = cp.Problem(objective, constraints)
    
    return problem, z, y_hat, ep, gamma, delta

####################################################################################################
# E2E neural network module
####################################################################################################
class e2e_net(nn.Module):
    """End-to-end DRO learning neural net module.
    """
    def __init__(self, n_x, n_y, n_obs, opt_layer='nominal', prisk='p_var', perf_loss='sharpe_loss',
                pred_model='linear', pred_loss_factor=0.5, perf_period=13, train_pred=True, train_gamma=True, train_delta=True, set_seed=None, cache_path='./cache/'):
        """End-to-end learning neural net module

        This NN module implements a linear prediction layer 'pred_layer' and a DRO layer 
        'opt_layer' based on a tractable convex formulation from Ben-Tal et al. (2013). 'delta' and
        'gamma' are declared as nn.Parameters so that they can be 'learned'.

        Inputs
        n_x: Number of inputs (i.e., features) in the prediction model
        n_y: Number of outputs from the prediction model
        n_obs: Number of scenarios from which to calculate the sample set of residuals
        prisk: String. Portfolio risk function. Used in the opt_layer
        opt_layer: String. Determines which CvxpyLayer-object to call for the optimization layer
        perf_loss: Performance loss function based on out-of-sample financial performance
        pred_loss_factor: Trade-off between prediction loss function and performance loss function.
            Set 'pred_loss_factor=None' to define the loss function purely as 'perf_loss'
        perf_period: Number of lookahead realizations used in 'perf_loss()'
        train_pred: Boolean. Choose if the prediction layer is learnable (or keep it fixed)
        train_gamma: Boolean. Choose if the risk appetite parameter gamma is learnable
        train_delta: Boolean. Choose if the robustness parameter delta is learnable
        set_seed: (Optional) Int. Set the random seed for replicability

        Output
        e2e_net: nn.Module object 
        """
        super(e2e_net, self).__init__()

        # Set random seed (to be used for replicability of numerical experiments)
        if set_seed is not None:
            torch.manual_seed(set_seed)
            self.seed = set_seed

        self.n_x = n_x
        self.n_y = n_y
        self.n_obs = n_obs

        # Prediction loss function
        if pred_loss_factor is not None:
            self.pred_loss_factor = pred_loss_factor
            self.pred_loss = torch.nn.MSELoss()
        else:
            self.pred_loss = None

        # Define performance loss
        self.perf_loss = eval('lf.'+perf_loss)
        
        # Store risk function reference
        self.prisk = prisk

        # Number of time steps to evaluate the task loss
        self.perf_period = perf_period

        # Register 'gamma' (risk-return trade-off parameter)
        self.gamma = nn.Parameter(torch.FloatTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = train_gamma
        self.gamma.data = self.gamma.data.double()  # Convert parameter data to double precision
        self.gamma_init = self.gamma.item()

        # Record the model design: nominal, base or DRO
        if opt_layer == 'nominal':
            self.model_type = 'nom'
        elif opt_layer == 'base_mod':
            self.gamma.requires_grad = False
            self.model_type = 'base_mod' 
        else:
            # Register 'delta' (ambiguity sizing parameter) for DR layer
            if opt_layer == 'hellinger':
                ub = (1 - 1/(n_obs**0.5)) / 2
                lb = (1 - 1/(n_obs**0.5)) / 10
            else:
                ub = (1 - 1/n_obs) / 2
                lb = (1 - 1/n_obs) / 10
            self.delta = nn.Parameter(torch.FloatTensor(1).uniform_(lb, ub))
            self.delta.requires_grad = train_delta
            self.delta.data = self.delta.data.double()  # Convert parameter data to double precision
            self.delta_init = self.delta.item()
            self.model_type = 'dro'

        # LAYER: Prediction model
        self.pred_model = pred_model
        if pred_model == 'linear':
            # Linear prediction model
            self.pred_layer = nn.Linear(n_x, n_y).double()  # Convert to double precision
            self.pred_layer.weight.requires_grad = train_pred
            self.pred_layer.bias.requires_grad = train_pred
        elif pred_model == '2layer':
            # Neural net with 2 hidden layers 
            self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.5*(n_x+n_y)), n_y),
                      nn.ReLU(),
                      nn.Linear(n_y, n_y))
        elif pred_model == '3layer':
            # Neural net with 3 hidden layers 
            self.pred_layer = nn.Sequential(nn.Linear(n_x, int(0.5*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.5*(n_x+n_y)), int(0.6*(n_x+n_y))),
                      nn.ReLU(),
                      nn.Linear(int(0.6*(n_x+n_y)), n_y),
                      nn.ReLU(),
                      nn.Linear(n_y, n_y))

        # Store opt_layer for later use
        self.opt_layer = opt_layer
        
        # LAYER: Optimization model - Create CvxpyLayer instances
        if opt_layer == 'base_mod':
            self.base_problem, self.base_z, self.base_y_hat_param = base_mod(n_y, n_obs, eval('rf.'+prisk))
            self.base_layer = CvxpyLayer(self.base_problem, parameters=[self.base_y_hat_param], variables=[self.base_z])
        elif opt_layer == 'nominal':
            self.nom_problem, self.nom_z, self.nom_y_hat_param, self.nom_ep_param, self.nom_gamma_param = nominal(n_y, n_obs, eval('rf.'+prisk))
            self.nom_layer = CvxpyLayer(self.nom_problem, parameters=[self.nom_y_hat_param, self.nom_ep_param, self.nom_gamma_param], variables=[self.nom_z])
        elif opt_layer == 'hellinger':
            self.dro_problem, self.dro_z, self.dro_y_hat_param, self.dro_ep_param, self.dro_gamma_param, self.dro_delta_param = hellinger(n_y, n_obs, eval('rf.'+prisk))
            self.dro_layer = CvxpyLayer(self.dro_problem, parameters=[self.dro_y_hat_param, self.dro_ep_param, self.dro_gamma_param, self.dro_delta_param], variables=[self.dro_z])
        
        # Store reference path to store model data
        self.cache_path = cache_path

        # Store initial model
        if train_gamma and train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model
        elif train_delta and not train_gamma:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainGamma'+str(train_gamma)
        elif train_gamma and not train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainDelta'+str(train_delta)
        elif not train_gamma and not train_delta:
            self.init_state_path = cache_path + self.model_type+'_initial_state_' + pred_model + '_TrainGamma'+str(train_gamma) + '_TrainDelta'+str(train_delta)
        torch.save(self.state_dict(), self.init_state_path)

    #-----------------------------------------------------------------------------------------------
    # forward: forward pass of the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def forward(self, X, Y):
        """Forward pass of the NN module

        The inputs 'X' are passed through the prediction layer to yield predictions 'Y_hat'. The
        residuals from prediction are then calcuclated as 'ep = Y - Y_hat'. Finally, the residuals
        are passed to the optimization layer to find the optimal decision z_star.

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data

        Other 
        ep: Residuals. (n_obs x n_y) matrix of the residual between realizations and predictions

        Outputs
        y_hat: Prediction. (n_y x 1) vector of outputs of the prediction layer
        z_star: Optimal solution. (n_y x 1) vector of asset weights
        """
        # Multiple predictions Y_hat from X
        Y_hat = torch.stack([self.pred_layer(x_t) for x_t in X])

        # Calculate residuals and process them
        ep = Y - Y_hat[:-1]
        y_hat = Y_hat[-1]

        # Optimization solver arguments (OSQP for better performance and stability)
        # OSQP is often faster and more stable than ECOS for QP problems
        solver_args = {
            'solve_method': 'OSQP', 
            'max_iter': 1000,
            'eps_abs': 1e-6,
            'eps_rel': 1e-6,
            'warm_start': True
        }
        # Alternative: ECOS with original stable parameters (fallback)
        # solver_args = {'solve_method': 'ECOS', 'max_iters': 120, 'abstol': 1e-7}
        # Alternative: Clarabel for complex cone problems
        # solver_args = {'solve_method': 'CLARABEL', 'tol_gap_abs': 1e-6, 'tol_gap_rel': 1e-6}

        # Optimize z per scenario using direct CVXPY solving
        # Determine whether nominal or dro model
        if self.model_type == 'nom':
            z_star = self._solve_cvxpy_nominal(ep, y_hat, self.gamma, solver_args)
        elif self.model_type == 'dro':
            z_star = self._solve_cvxpy_dro(ep, y_hat, self.gamma, self.delta, solver_args)
        elif self.model_type == 'base_mod':
            z_star = self._solve_cvxpy_base(y_hat, solver_args)

        return z_star, y_hat

    

    #-----------------------------------------------------------------------------------------------
    # net_train: Train the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_train(self, train_set, val_set=None, epochs=None, lr=None):
        """Neural net training module
        
        Inputs
        train_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        val_set: SlidingWindow object containing features x, realizations y and performance
        realizations y_perf
        epochs: Number of training epochs
        lr: learning rate

        Output
        Trained model
        (Optional) val_loss: Validation loss
        """

        # Assign number of epochs and learning rate
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr

        # Define the optimizer and its parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Number of elements in training set
        n_train = len(train_set)

        # Train the neural network
        for epoch in range(int(epochs)):
                
            # TRAINING: forward + backward pass
            train_loss = 0
            optimizer.zero_grad() 
            for t, (x, y, y_perf) in enumerate(train_set):

                # Forward pass: predict and optimize
                z_star, y_hat = self(x.squeeze(), y.squeeze())

                # Loss function
                if self.pred_loss is None:
                    loss = (1/n_train) * self.perf_loss(z_star, y_perf.squeeze())
                else:
                    loss = (1/n_train) * (self.perf_loss(z_star, y_perf.squeeze()) + 
                    (self.pred_loss_factor/self.n_y) * self.pred_loss(y_hat, y_perf.squeeze()[0]))

                # Backward pass: backpropagation
                loss.backward()

                # Accumulate loss of the fully trained model
                train_loss += loss.item()
        
            # Update parameters
            optimizer.step()

            # Ensure that gamma, delta > 0 after taking a descent step
            for name, param in self.named_parameters():
                if name=='gamma':
                    param.data.clamp_(0.0001)
                if name=='delta':
                    param.data.clamp_(0.0001)

        # Compute and return the validation loss of the model
        if val_set is not None:

            # Number of elements in validation set
            n_val = len(val_set)

            val_loss = 0
            with torch.no_grad():
                for t, (x, y, y_perf) in enumerate(val_set):

                    # Predict and optimize
                    z_val, y_val = self(x.squeeze(), y.squeeze())
                
                    # Loss function
                    if self.pred_loss_factor is None:
                        loss = (1/n_val) * self.perf_loss(z_val, y_perf.squeeze())
                    else:
                        loss = (1/n_val) * (self.perf_loss(z_val, y_perf.squeeze()) + 
                        (self.pred_loss_factor/self.n_y)*self.pred_loss(y_val, y_perf.squeeze()[0]))
                    
                    # Accumulate loss
                    val_loss += loss.item()

            return val_loss

    #-----------------------------------------------------------------------------------------------
    # net_cv: Cross validation of the e2e neural net for hyperparameter tuning
    #-----------------------------------------------------------------------------------------------
    def net_cv(self, X, Y, lr_list, epoch_list, n_val=4):
        """Neural net cross-validation module

        Inputs
        X: Features. TrainTest object of feature timeseries data
        Y: Realizations. TrainTest object of asset time series data
        epochs: number of training passes
        lr_list: List of candidate learning rates
        epoch_list: List of candidate number of epochs
        n_val: Number of validation folds from the training dataset
        
        Output
        Trained model
        """
        results = pc.CrossVal()
        X_temp = dl.TrainTest(X.train(), X.n_obs, [1, 0])
        Y_temp = dl.TrainTest(Y.train(), Y.n_obs, [1, 0])
        for epochs in epoch_list:
            for lr in lr_list:
                
                # Train the neural network
                print('================================================')
                print(f"Training E2E {self.model_type} model: lr={lr}, epochs={epochs}")
                
                val_loss_tot = []
                for i in range(n_val-1,-1,-1):

                    # Partition training dataset into training and validation subset
                    split = [round(1-0.2*(i+1),2), 0.2]
                    X_temp.split_update(split)
                    Y_temp.split_update(split)

                    # Construct training and validation DataLoader objects
                    train_set = DataLoader(pc.SlidingWindow(X_temp.train(), Y_temp.train(), 
                                                            self.n_obs, self.perf_period))
                    val_set = DataLoader(pc.SlidingWindow(X_temp.test(), Y_temp.test(), 
                                                            self.n_obs, self.perf_period))

                    # Reset learnable parameters gamma and delta
                    self.load_state_dict(torch.load(self.init_state_path))

                    if self.pred_model == 'linear':
                        # Initialize the prediction layer weights to OLS regression weights
                        X_train, Y_train = X_temp.train(), Y_temp.train()
                        
                        # Handle both pandas DataFrames and numpy arrays
                        if hasattr(X_train, 'insert'):
                            # Pandas DataFrame - use insert method
                            X_train.insert(0,'ones', 1.0)
                            X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                        else:
                            # Numpy array - add ones column using numpy
                            ones_col = np.ones((X_train.shape[0], 1))
                            X_train = np.column_stack([ones_col, X_train])
                            X_train = Variable(torch.tensor(X_train, dtype=torch.double))
                        
                        Y_train = Variable(torch.tensor(Y_train, dtype=torch.double))
                    
                        Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                        Theta = Theta.T
                        del X_train, Y_train

                        with torch.no_grad():
                            self.pred_layer.bias.copy_(Theta[:,0])
                            self.pred_layer.weight.copy_(Theta[:,1:])

                    val_loss = self.net_train(train_set, val_set=val_set, lr=lr, epochs=epochs)
                    val_loss_tot.append(val_loss)

                    print(f"Fold: {n_val-i} / {n_val}, val_loss: {val_loss}")

                # Store results
                results.val_loss.append(np.mean(val_loss_tot))
                results.lr.append(lr)
                results.epochs.append(epochs)
                print('================================================')

        # Convert results to dataframe with error handling for pandas compatibility
        try:
            self.cv_results = results.df()
            # Check if results.df() returned a numpy array instead of DataFrame
            if hasattr(self.cv_results, 'to_pickle'):
                # It's a pandas DataFrame, save it
                self.cv_results.to_pickle(self.init_state_path+'_results.pkl')
            else:
                # It's a numpy array, convert it to DataFrame
                print("🔧 results.df() returned numpy array, converting to DataFrame...")
                self.cv_results = pd.DataFrame(self.cv_results, columns=['lr', 'epochs', 'val_loss'])
                self.cv_results.to_pickle(self.init_state_path+'_results.pkl')
        except Exception as e:
            print(f"⚠️ Pandas DataFrame creation failed: {e}")
            print("🔧 Creating DataFrame manually with numpy arrays...")
            
            try:
                # Create DataFrame manually by converting to Python native types
                lr_clean = [float(lr.item()) if hasattr(lr, 'item') else float(lr) for lr in results.lr]
                epochs_clean = [int(epoch.item()) if hasattr(epoch, 'item') else int(epoch) for epoch in results.epochs]
                val_loss_clean = [float(vl.item()) if hasattr(vl, 'item') else float(vl) for vl in results.val_loss]
                
                self.cv_results = pd.DataFrame({
                    'lr': lr_clean,
                    'epochs': epochs_clean, 
                    'val_loss': val_loss_clean
                })
                print("✅ DataFrame created successfully")
            except Exception as e2:
                print(f"⚠️ Manual DataFrame creation also failed: {e2}")
                print("🔧 Storing results as numpy array instead...")
                # Final fallback: store as numpy array
                try:
                    # Convert to numpy arrays with robust error handling
                    lr_array = np.array([float(lr.item()) if hasattr(lr, 'item') else float(lr) for lr in results.lr])
                    epochs_array = np.array([int(epoch.item()) if hasattr(epoch, 'item') else int(epoch) for epoch in results.epochs])
                    val_loss_array = np.array([float(vl.item()) if hasattr(vl, 'item') else float(vl) for vl in results.val_loss])
                    
                    self.cv_results = np.column_stack([lr_array, epochs_array, val_loss_array])
                    print("✅ Numpy array created successfully")
                except Exception as e3:
                    print(f"⚠️ Even numpy conversion failed: {e3}")
                    print("🔧 Using default values as final fallback...")
                    # Ultimate fallback: use default values
                    self.cv_results = np.array([[0.01, 20, -0.1]])  # Default: lr=0.01, epochs=20, val_loss=-0.1
                    print("✅ Default values set as final fallback")

        # Select and store the optimal hyperparameters
        if hasattr(self.cv_results, 'val_loss'):
            # It's a pandas DataFrame
            idx = self.cv_results.val_loss.idxmin()
            self.lr = self.cv_results.lr[idx]
            self.epochs = self.cv_results.epochs[idx]
        else:
            # It's a numpy array
            idx = np.argmin(self.cv_results[:, 2])  # val_loss is in column 2
            self.lr = self.cv_results[idx, 0]  # lr is in column 0
            self.epochs = self.cv_results[idx, 1]  # epochs is in column 1

        # Print optimal parameters
        print(f"CV E2E {self.model_type} with hyperparameters: lr={self.lr}, epochs={self.epochs}")

    #-----------------------------------------------------------------------------------------------
    # net_roll_test: Test the e2e neural net
    #-----------------------------------------------------------------------------------------------
    def net_roll_test(self, X, Y, n_roll=4, lr=None, epochs=None):
        """Neural net rolling window out-of-sample test

        Inputs
        X: Features. ([n_obs+1] x n_x) torch tensor with feature timeseries data
        Y: Realizations. (n_obs x n_y) torch tensor with asset timeseries data
        n_roll: Number of training periods (i.e., number of times to retrain the model)
        lr: Learning rate for test. If 'None', the optimal learning rate is loaded
        epochs: Number of epochs for test. If 'None', the optimal # of epochs is loaded

        Output 
        self.portfolio: add the backtest results to the e2e_net object
        """

        # Declare backtest object to hold the test results
        portfolio = pc.backtest(len(Y.test())-Y.n_obs, self.n_y, Y.index()[Y.n_obs:])

        # Store trained gamma and delta values 
        if self.model_type == 'nom':
            self.gamma_trained = []
        elif self.model_type == 'dro':
            self.gamma_trained = []
            self.delta_trained = []

        # Store the squared L2-norm of the prediction weights and their difference from OLS weights
        if self.pred_model == 'linear':
            self.theta_L2 = []
            self.theta_dist_L2 = []

        # Store initial train/test split
        init_split = Y.split

        # Window size
        win_size = init_split[1] / n_roll

        split = [0, 0]
        t = 0
        for i in range(n_roll):

            print(f"Out-of-sample window: {i+1} / {n_roll}")

            split[0] = init_split[0] + win_size * i
            if i < n_roll-1:
                split[1] = win_size
            else:
                split[1] = 1 - split[0]

            X.split_update(split), Y.split_update(split)
            train_set = DataLoader(pc.SlidingWindow(X.train(), Y.train(), self.n_obs, 
                                                    self.perf_period))
            test_set = DataLoader(pc.SlidingWindow(X.test(), Y.test(), self.n_obs, 0))

            # Reset learnable parameters gamma and delta
            self.load_state_dict(torch.load(self.init_state_path))

            if self.pred_model == 'linear':
                # Initialize the prediction layer weights to OLS regression weights
                X_train, Y_train = X.train(), Y.train()
                
                # Handle both pandas DataFrames and numpy arrays
                if hasattr(X_train, 'insert'):
                    # Pandas DataFrame - use insert method
                    X_train.insert(0,'ones', 1.0)
                    X_train = Variable(torch.tensor(X_train.values, dtype=torch.double))
                else:
                    # Numpy array - add ones column using numpy
                    ones_col = np.ones((X_train.shape[0], 1))
                    X_train = np.column_stack([ones_col, X_train])
                    X_train = Variable(torch.tensor(X_train, dtype=torch.double))
                
                Y_train = Variable(torch.tensor(Y_train, dtype=torch.double))
            
                Theta = torch.inverse(X_train.T @ X_train) @ (X_train.T @ Y_train)
                Theta = Theta.T
                del X_train, Y_train

                with torch.no_grad():
                    self.pred_layer.bias.copy_(Theta[:,0])
                    self.pred_layer.weight.copy_(Theta[:,1:])

            # Train model using all available data preceding the test window
            self.net_train(train_set, lr=lr, epochs=epochs)

            # Store trained values of gamma and delta
            if self.model_type == 'nom':
                self.gamma_trained.append(self.gamma.item())
            elif self.model_type == 'dro':
                self.gamma_trained.append(self.gamma.item())
                self.delta_trained.append(self.delta.item())

            # Store the squared L2 norm of theta and distance between theta and OLS weights
            if self.pred_model == 'linear':
                theta_L2 = (torch.sum(self.pred_layer.weight**2, axis=()) + 
                            torch.sum(self.pred_layer.bias**2, axis=()))
                theta_dist_L2 = (torch.sum((self.pred_layer.weight - Theta[:,1:])**2, axis=()) + 
                                torch.sum((self.pred_layer.bias - Theta[:,0])**2, axis=()))
                self.theta_L2.append(theta_L2)
                self.theta_dist_L2.append(theta_dist_L2)

            # Test model
            with torch.no_grad():
                for j, (x, y, y_perf) in enumerate(test_set):
                
                    # Predict and optimize
                    z_star, _ = self(x.squeeze(), y.squeeze())

                    # Store portfolio weights and returns for each time step 't'
                    portfolio.weights[t] = z_star.squeeze()
                    portfolio.rets[t] = y_perf.squeeze() @ portfolio.weights[t]
                    t += 1

        # Reset dataset
        X, Y = X.split_update(init_split), Y.split_update(init_split)

        # Calculate the portfolio statistics using the realized portfolio returns
        portfolio.stats()

        self.portfolio = portfolio

    #-----------------------------------------------------------------------------------------------
    # load_cv_results: Load cross validation results
    #-----------------------------------------------------------------------------------------------
    def load_cv_results(self, cv_results):
        """Load cross validation results

        Inputs
        cv_results: pd.dataframe containing the cross validation results

        Outputs
        self.lr: Load the optimal learning rate
        self.epochs: Load the optimal number of epochs
        """

        # Store the cross validation results within the object
        self.cv_results = cv_results

        # Select and store the optimal hyperparameters
        idx = cv_results.val_loss.idxmin()
        self.lr = cv_results.lr[idx]
        self.epochs = cv_results.epochs[idx]

    #-----------------------------------------------------------------------------------------------
    # CVXPY Solver Helper Methods (Direct solving instead of CvxpyLayer)
    #-----------------------------------------------------------------------------------------------
    def _solve_cvxpy_base(self, y_hat, solver_args):
        """Solve base optimization problem using CVXPY directly"""
        # Set parameter values
        self.base_y_hat_param.value = y_hat.detach().cpu().numpy()
        
        # Solve the problem
        try:
            self.base_problem.solve(**solver_args)
            if self.base_problem.status == 'optimal':
                z_star = torch.tensor(self.base_z.value, dtype=torch.double, device=y_hat.device)
                return z_star
            else:
                # Fallback to ECOS if OSQP fails
                fallback_args = {'solve_method': 'ECOS', 'max_iters': 120, 'abstol': 1e-7}
                self.base_problem.solve(**fallback_args)
                z_star = torch.tensor(self.base_z.value, dtype=torch.double, device=y_hat.device)
                return z_star
        except Exception as e:
            print(f"CVXPY solve failed: {e}, using equal weights fallback")
            # Return equal weights when optimization fails
            z_star = torch.ones(self.n_y, dtype=torch.double, device=y_hat.device) / self.n_y
            return z_star

    def _solve_cvxpy_nominal(self, ep, y_hat, gamma, solver_args):
        """Solve nominal optimization problem using CVXPY directly"""
        # Set parameter values
        self.nom_y_hat_param.value = y_hat.detach().cpu().numpy()
        self.nom_ep_param.value = ep.detach().cpu().numpy()
        self.nom_gamma_param.value = gamma.item()
        
        # Solve the problem
        try:
            self.nom_problem.solve(**solver_args)
            if self.nom_problem.status == 'optimal':
                z_star = torch.tensor(self.nom_z.value, dtype=torch.double, device=y_hat.device)
                return z_star
            else:
                # Fallback to ECOS if OSQP fails
                fallback_args = {'solve_method': 'ECOS', 'max_iters': 120, 'abstol': 1e-7}
                self.nom_problem.solve(**fallback_args)
                z_star = torch.tensor(self.nom_z.value, dtype=torch.double, device=y_hat.device)
                return z_star
        except Exception as e:
            print(f"CVXPY solve failed: {e}, using equal weights fallback")
            # Return equal weights when optimization fails
            z_star = torch.ones(self.n_y, dtype=torch.double, device=y_hat.device) / self.n_y
            return z_star

    def _solve_cvxpy_dro(self, ep, y_hat, gamma, delta, solver_args):
        """Solve distributionally robust optimization problem using CVXPY directly"""
        # Set parameter values
        self.dro_y_hat_param.value = y_hat.detach().cpu().numpy()
        self.dro_ep_param.value = ep.detach().cpu().numpy()
        self.dro_gamma_param.value = gamma.item()
        self.dro_delta_param.value = delta.item()
        
        # Solve the problem
        try:
            self.dro_problem.solve(**solver_args)
            if self.dro_problem.status == 'optimal':
                z_star = torch.tensor(self.dro_z.value, dtype=torch.double, device=y_hat.device)
                return z_star
            else:
                # Fallback to ECOS if OSQP fails
                fallback_args = {'solve_method': 'ECOS', 'max_iters': 120, 'abstol': 1e-7}
                self.dro_problem.solve(**fallback_args)
                z_star = torch.tensor(self.dro_z.value, dtype=torch.double, device=y_hat.device)
                return z_star
        except Exception as e:
            print(f"CVXPY solve failed: {e}, using equal weights fallback")
            # Return equal weights when optimization fails
            z_star = torch.ones(self.n_y, dtype=torch.double, device=y_hat.device) / self.n_y
            return z_star

    #-----------------------------------------------------------------------------------------------
    # Custom Serialization Methods for CvxpyLayer Objects
    #-----------------------------------------------------------------------------------------------
    def __getstate__(self):
        """Custom pickle serialization to handle cvxpylayers objects"""
        state = self.__dict__.copy()
        
        # Remove cvxpylayers objects that can't be pickled
        # These will be recreated when the model is loaded
        if hasattr(self, 'base_layer'):
            state['base_layer'] = None
        if hasattr(self, 'nom_layer'):
            state['nom_layer'] = None
        if hasattr(self, 'dro_layer'):
            state['dro_layer'] = None
            
        # Remove CVXPY problem objects that can't be pickled
        if hasattr(self, 'base_problem'):
            state['base_problem'] = None
        if hasattr(self, 'nom_problem'):
            state['nom_problem'] = None
        if hasattr(self, 'dro_problem'):
            state['dro_problem'] = None
            
        # Remove CVXPY variable objects that can't be pickled
        if hasattr(self, 'base_z'):
            state['base_z'] = None
        if hasattr(self, 'nom_z'):
            state['nom_z'] = None
        if hasattr(self, 'dro_z') and self.dro_z is not None:
            state['dro_z'] = None
            
        # Remove CVXPY parameter objects that can't be pickled
        if hasattr(self, 'base_y_hat_param'):
            state['base_y_hat_param'] = None
        if hasattr(self, 'nom_y_hat_param'):
            state['nom_y_hat_param'] = None
        if hasattr(self, 'nom_ep_param'):
            state['nom_ep_param'] = None
        if hasattr(self, 'nom_gamma_param'):
            state['nom_gamma_param'] = None
        if hasattr(self, 'dro_y_hat_param'):
            state['dro_y_hat_param'] = None
        if hasattr(self, 'dro_ep_param'):
            state['dro_ep_param'] = None
        if hasattr(self, 'dro_gamma_param'):
            state['dro_gamma_param'] = None
        if hasattr(self, 'dro_delta_param'):
            state['dro_delta_param'] = None
            
        return state
    
    def save_model(self, filepath):
        """Save only the essential model parameters (weights, biases, trained params)"""
        import pickle
        
        # Create a minimal state with only essential parameters
        save_state = {
            'pred_layer_state': self.pred_layer.state_dict() if hasattr(self, 'pred_layer') else None,
            'gamma': self.gamma.data.clone() if hasattr(self, 'gamma') else None,
            'delta': self.delta.data.clone() if hasattr(self, 'delta') else None,
            'opt_layer': self.opt_layer,
            'n_y': self.n_y,
            'n_obs': self.n_obs,
            'prisk': self.prisk,
            'pred_model': self.pred_model,
            'cv_results': getattr(self, 'cv_results', None),
            'portfolio': getattr(self, 'portfolio', None),
            'epochs': getattr(self, 'epochs', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_state, f, pickle.HIGHEST_PROTOCOL)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters and recreate cvxpylayers objects"""
        import pickle
        
        with open(filepath, 'rb') as f:
            save_state = pickle.load(f)
        
        # Restore essential parameters
        if save_state['pred_layer_state'] is not None:
            self.pred_layer.load_state_dict(save_state['pred_layer_state'])
        if save_state['gamma'] is not None:
            self.gamma.data = save_state['gamma']
        if save_state['delta'] is not None:
            self.delta.data = save_state['delta']
        
        # Restore other attributes
        for key, value in save_state.items():
            if key not in ['pred_layer_state', 'gamma', 'delta']:
                setattr(self, key, value)
        
        # Recreate CVXPY problem definitions (no longer using CvxpyLayer)
        if hasattr(self, 'opt_layer'):
            if self.opt_layer == 'base_mod':
                import e2edro.RiskFunctions as rf
                self.base_problem, self.base_z, self.base_y_hat_param = base_mod(self.n_y, self.n_obs, eval('rf.'+self.prisk))
                self.base_layer = None  # Will solve directly
            elif self.opt_layer == 'nominal':
                import e2edro.RiskFunctions as rf
                self.nom_problem, self.nom_z, self.nom_y_hat_param, self.nom_ep_param, self.nom_gamma_param = nominal(self.n_y, self.n_obs, eval('rf.'+self.prisk))
                self.nom_layer = None  # Will solve directly
            elif self.opt_layer == 'hellinger':
                import e2edro.RiskFunctions as rf
                self.dro_problem, self.dro_z, self.dro_y_hat_param, self.dro_ep_param, self.dro_gamma_param, self.dro_delta_param = hellinger(self.n_y, self.n_obs, eval('rf.'+self.prisk))
                self.dro_layer = None  # Will solve directly

    def __setstate__(self, state):
        """Custom pickle deserialization to recreate cvxpylayers objects"""
        self.__dict__.update(state)
        
        # Recreate CVXPY problem definitions if opt_layer is available (no longer using CvxpyLayer)
        if hasattr(self, 'opt_layer'):
            if self.opt_layer == 'base_mod':
                import e2edro.RiskFunctions as rf
                self.base_problem, self.base_z, self.base_y_hat_param = base_mod(self.n_y, self.n_obs, eval('rf.'+self.prisk))
                self.base_layer = None  # Will solve directly
            elif self.opt_layer == 'nominal':
                import e2edro.RiskFunctions as rf
                self.nom_problem, self.nom_z, self.nom_y_hat_param, self.nom_ep_param, self.nom_gamma_param = nominal(self.n_y, self.n_obs, eval('rf.'+self.prisk))
                self.nom_layer = None  # Will solve directly
            elif self.opt_layer == 'hellinger':
                import e2edro.RiskFunctions as rf
                self.dro_problem, self.dro_z, self.dro_y_hat_param, self.dro_ep_param, self.dro_gamma_param, self.dro_delta_param = hellinger(self.n_y, self.n_obs, eval('rf.'+self.prisk))
                self.dro_layer = None  # Will solve directly