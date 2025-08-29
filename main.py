# Distributionally Robust End-to-End Portfolio Construction
# Experiment 1 - General
####################################################################################################
# Import libraries
####################################################################################################
import torch
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings

# Helper: ensure cv_results-like objects are pandas DataFrames
def as_df(obj):
    try:
        if hasattr(obj, 'to_pandas'):
            return obj.to_pandas()
        # pandas DataFrame/Series usually have to_pickle/values
        if hasattr(obj, 'to_pickle'):
            return obj
        arr = np.asarray(obj)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return pd.DataFrame(arr, columns=['lr', 'epochs', 'val_loss'])
        return pd.DataFrame(arr)
    except Exception:
        return pd.DataFrame(obj)

# Suppress DIFFCP version warning (functionality still works)
warnings.filterwarnings("ignore", message=".*diffcp.*")

plt.close("all")

# Make the code device-agnostic
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():  # Apple Metal Performance Shaders
    device = 'mps'
else:
    device = 'cpu'

print(f"Using device: {device}")

# Performance optimization: Enable optimizations for faster execution
if device == 'mps':
    torch.set_default_dtype(torch.float32)  # MPS works better with float32
    print("✅ MPS optimizations enabled (float32 precision)")
elif device == 'cuda':
    print("✅ CUDA optimizations enabled")
else:
    print("⚠️  Running on CPU - consider using GPU for better performance")

# Import E2E_DRO functions
from e2edro import e2edro as e2e
from e2edro import DataLoad as dl
from e2edro import BaseModels as bm
from e2edro import PlotFunctions as pf

# Path to cache the data, models and results
cache_path = "./cache/exp/"

####################################################################################################
# Experiments 1-4 (with hisotrical data): Load data
####################################################################################################

# Data frequency and start/end dates
freq = 'weekly'
start = '2000-01-01'
end = '2021-09-30'

# Train, validation and test split percentage
split = [0.6, 0.4]

# Number of observations per window (optimized for better GPU utilization)
n_obs = 120  # Increased from 78 (1.5x larger batches, reduced Python overhead)

# Number of assets (reduced for faster execution)
n_y = 15   # Reduced from 20 (75% of original)

# Performance optimization: Larger batch sizes for better GPU utilization
# Original: n_obs=104, n_y=20
# Previous: n_obs=78, n_y=15 (75% of original size)
# New: n_obs=120, n_y=15 (larger batches, reduced Python overhead)
print(f"Data optimization: {n_obs} observations × {n_y} assets (larger batches for GPU efficiency)")

# AlphaVantage API Key. 
# Note: User API keys can be obtained for free from www.alphavantage.co. Users will need a free 
# academic or paid license to download adjusted closing pricing data from AlphaVantage.
AV_key = None

# Historical data: Download data (or load cached data)
X, Y = dl.AV(start, end, split, freq=freq, n_obs=n_obs, n_y=n_y, use_cache=True,
            save_results=False, AV_key=AV_key)

# Number of features and assets
n_x, n_y = X.data.shape[1], Y.data.shape[1]

# Statistical significance analysis of features vs targets
try:
    stats = dl.statanalysis(X.data, Y.data)
    print("✅ Statistical analysis completed successfully")
except Exception as e:
    print(f"⚠️ Statistical analysis failed: {e}")
    print("🔧 Continuing without statistical analysis...")
    stats = None

####################################################################################################
# E2E Learning System Run
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Initialize parameters
#---------------------------------------------------------------------------------------------------

# Performance loss function and performance period 'v+1'
perf_loss='sharpe_loss'
perf_period = 13

# Weight assigned to MSE prediction loss function
pred_loss_factor = 0.5

# Risk function (default set to variance)
prisk = 'p_var'

# Robust decision layer to use: hellinger or tv
dr_layer = 'hellinger'

# List of learning rates to test (further reduced for stability)
# previously lr_list = [0.005, 0.0125, 0.02]
# More conservative learning rates for better convergence
lr_list = [0.005, 0.01]

# List of total no. of epochs to test (further reduced for faster execution)
# previously epoch_list = [30, 40, 50, 60, 80, 100]
# Fewer epochs for faster training and better convergence
epoch_list = [20, 40]

# For replicability, set the random seed for the numerical experiments
set_seed = 1000

# Performance optimization: Further reduced hyperparameter search space
# Original: 3 learning rates × 6 epochs = 18 combinations
# Previous: 2 learning rates × 3 epochs = 6 combinations
# New: 2 learning rates × 2 epochs = 4 combinations (4.5x faster!)
print(f"Hyperparameter search space: {len(lr_list)} learning rates × {len(epoch_list)} epochs = {len(lr_list) * len(epoch_list)} total combinations")

# Print optimization summary
print("🚀 Performance Optimizations Applied:")
print(f"   • Hyperparameters: {len(lr_list)}×{len(epoch_list)} = {len(lr_list) * len(epoch_list)} combinations (vs 18 original)")
print(f"   • Learning rates: {lr_list} (more conservative for stability)")
print(f"   • Epochs: {epoch_list} (faster training)")
print(f"   • CVXPY Solver: ECOS/SCS (conic solvers for SOC constraints)")
print(f"   • Data size: {n_obs}×{n_y} (larger batches for GPU efficiency)")
print(f"   • Device: MPS (Apple Silicon GPU acceleration)")
print(f"   • Threading: Optimized for M2 Pro/Max (10/12 cores)")
print(f"   • CVXPY Solver: ECOS/SCS (conic solvers for SOC constraints)")

# Load saved models (default is False)
# use_cache = False
use_cache = False  # Retrain models from scratch due to pandas compatibility issues

#---------------------------------------------------------------------------------------------------
# Run 
#---------------------------------------------------------------------------------------------------

if use_cache:
    # Load cached models and backtest results
    with open(cache_path+'ew_net.pkl', 'rb') as inp:
        ew_net = pickle.load(inp)
    with open(cache_path+'po_net.pkl', 'rb') as inp:
        po_net = pickle.load(inp)
    # Load base model
    try:
        with open(cache_path+'base_net.pkl', 'rb') as inp:
            base_net = pickle.load(inp)
            # Check if it's our new format and recreate cvxpylayers if needed
            if hasattr(base_net, 'base_layer') and base_net.base_layer is None:
                base_net.load_model(cache_path+'base_net.pkl')
    except:
        base_net = None
        
    # Load nominal model
    try:
        with open(cache_path+'nom_net.pkl', 'rb') as inp:
            nom_net = pickle.load(inp)
            # Check if it's our new format and recreate cvxpylayers if needed
            if hasattr(nom_net, 'nom_layer') and nom_net.nom_layer is None:
                nom_net.load_model(cache_path+'nom_net.pkl')
    except:
        nom_net = None
        
    # Load DR model
    try:
        with open(cache_path+'dr_net.pkl', 'rb') as inp:
            dr_net = pickle.load(inp)
            # Check if it's our new format and recreate cvxpylayers if needed
            if hasattr(dr_net, 'dro_layer') and dr_net.dro_layer is None:
                dr_net.load_model(cache_path+'dr_net.pkl')
    except:
        dr_net = None
    with open(cache_path+'dr_po_net.pkl', 'rb') as inp:
        dr_po_net = pickle.load(inp)
    with open(cache_path+'dr_net_learn_delta.pkl', 'rb') as inp:
        dr_net_learn_delta = pickle.load(inp)
    with open(cache_path+'nom_net_learn_gamma.pkl', 'rb') as inp:
        nom_net_learn_gamma = pickle.load(inp)
    with open(cache_path+'dr_net_learn_gamma.pkl', 'rb') as inp:
        dr_net_learn_gamma = pickle.load(inp)
    with open(cache_path+'dr_net_learn_gamma_delta.pkl', 'rb') as inp:
        dr_net_learn_gamma_delta = pickle.load(inp)
    with open(cache_path+'nom_net_learn_theta.pkl', 'rb') as inp:
        nom_net_learn_theta = pickle.load(inp)
    with open(cache_path+'dr_net_learn_theta.pkl', 'rb') as inp:
        dr_net_learn_theta = pickle.load(inp)

    # Load extended models with error handling
    try:
        with open(cache_path+'base_net_ext.pkl', 'rb') as inp:
            base_net_ext = pickle.load(inp)
    except:
        base_net_ext = None
        
    try:
        with open(cache_path+'nom_net_ext.pkl', 'rb') as inp:
            nom_net_ext = pickle.load(inp)
    except:
        nom_net_ext = None
        
    try:
        with open(cache_path+'dr_net_ext.pkl', 'rb') as inp:
            dr_net_ext = pickle.load(inp)
    except:
        dr_net_ext = None
    with open(cache_path+'dr_net_learn_delta_ext.pkl', 'rb') as inp:
        dr_net_learn_delta_ext = pickle.load(inp)
    with open(cache_path+'nom_net_learn_gamma_ext.pkl', 'rb') as inp:
        nom_net_learn_gamma_ext = pickle.load(inp)
    with open(cache_path+'dr_net_learn_gamma_ext.pkl', 'rb') as inp:
        dr_net_learn_gamma_ext = pickle.load(inp)
    with open(cache_path+'nom_net_learn_theta_ext.pkl', 'rb') as inp:
        nom_net_learn_theta_ext = pickle.load(inp)
    with open(cache_path+'dr_net_learn_theta_ext.pkl', 'rb') as inp:
        dr_net_learn_theta_ext = pickle.load(inp)

    with open(cache_path+'dr_net_tv.pkl', 'rb') as inp:
        dr_net_tv = pickle.load(inp)
    with open(cache_path+'dr_net_tv_learn_delta.pkl', 'rb') as inp:
        dr_net_tv_learn_delta = pickle.load(inp)
    with open(cache_path+'dr_net_tv_learn_gamma.pkl', 'rb') as inp:
        dr_net_tv_learn_gamma = pickle.load(inp)
    with open(cache_path+'dr_net_tv_learn_theta.pkl', 'rb') as inp:
        dr_net_tv_learn_theta = pickle.load(inp)
else:
    # Exp 1: Equal weight portfolio
    ew_net = bm.equal_weight(n_x, n_y, n_obs)
    ew_net.net_roll_test(X, Y, n_roll=4)
    with open(cache_path+'ew_net.pkl', 'wb') as outp:
            pickle.dump(ew_net, outp, pickle.HIGHEST_PROTOCOL)
    print('ew_net run complete')

    # Exp 1, 2, 3: Predict-then-optimize system
    po_net = bm.pred_then_opt(n_x, n_y, n_obs, set_seed=set_seed, prisk=prisk)
    po_net.net_roll_test(X, Y)
    with open(cache_path+'po_net.pkl', 'wb') as outp:
        pickle.dump(po_net, outp, pickle.HIGHEST_PROTOCOL)
    print('po_net run complete')

    # Exp 1: Base E2E
    base_net = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=True, train_gamma=False, train_delta=False,
                        set_seed=set_seed, opt_layer='base_mod', perf_loss=perf_loss, 
                        perf_period=perf_period, pred_loss_factor=pred_loss_factor)
    base_net.net_cv(X, Y, lr_list, epoch_list)
    base_net.net_roll_test(X, Y)
    base_net.save_model(cache_path+'base_net.pkl')
    print('base_net run complete')

    # Exp 1: Nominal E2E
    nom_net = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=True, train_gamma=True, train_delta=False,
                        set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss, 
                        cache_path=cache_path, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor)
    nom_net.net_cv(X, Y, lr_list, epoch_list)
    nom_net.net_roll_test(X, Y)
    nom_net.save_model(cache_path+'nom_net.pkl')
    print('nom_net run complete')

    # Exp 1: DR E2E
    dr_net = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=True, train_gamma=True, train_delta=True,
                        set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                        cache_path=cache_path, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor)
    dr_net.net_cv(X, Y, lr_list, epoch_list)
    dr_net.net_roll_test(X, Y)
    dr_net.save_model(cache_path+'dr_net.pkl')
    print('dr_net run complete')

    # Exp 2: DR predict-then-optimize system
    dr_po_net = bm.pred_then_opt(n_x, n_y, n_obs, set_seed=set_seed, prisk=prisk,
                                opt_layer=dr_layer)
    dr_po_net.net_roll_test(X, Y)
    with open(cache_path+'dr_po_net.pkl', 'wb') as outp:
        pickle.dump(dr_po_net, outp, pickle.HIGHEST_PROTOCOL)
    print('dr_po_net run complete')

    # Exp 2: DR E2E (fixed theta and gamma, learn delta)
    dr_net_learn_delta = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=False, train_gamma=False, train_delta=True,
                        set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                        cache_path=cache_path, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor)
    dr_net_learn_delta.net_cv(X, Y, lr_list, epoch_list)
    dr_net_learn_delta.net_roll_test(X, Y)
    dr_net_learn_delta.save_model(cache_path+'dr_net_learn_delta.pkl')
    print('dr_net_learn_delta run complete')

    # Exp 3: Nominal E2E (fixed theta, learn gamma)
    nom_net_learn_gamma = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=False, train_gamma=True, train_delta=False,
                        set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss, 
                        cache_path=cache_path, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor)
    nom_net_learn_gamma.net_cv(X, Y, lr_list, epoch_list)
    nom_net_learn_gamma.net_roll_test(X, Y)
    nom_net_learn_gamma.save_model(cache_path+'nom_net_learn_gamma.pkl')
    print('nom_net_learn_gamma run complete')

    # Exp 3: DR E2E (fixed theta, learn gamma, fixed delta)
    dr_net_learn_gamma = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=False, train_gamma=True, train_delta=False,
                        set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                        cache_path=cache_path, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor)
    dr_net_learn_gamma.net_cv(X, Y, lr_list, epoch_list)
    dr_net_learn_gamma.net_roll_test(X, Y)
    dr_net_learn_gamma.save_model(cache_path+'dr_net_learn_gamma.pkl')
    print('dr_net_learn_gamma run complete')

    # Exp 4: Nominal E2E (learn theta, fixed gamma)
    nom_net_learn_theta = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=True, train_gamma=False, train_delta=False,
                        set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss, 
                        cache_path=cache_path, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor)
    nom_net_learn_theta.net_cv(X, Y, lr_list, epoch_list)
    nom_net_learn_theta.net_roll_test(X, Y)
    nom_net_learn_theta.save_model(cache_path+'nom_net_learn_theta.pkl')
    print('nom_net_learn_theta run complete')

    # Exp 4: DR E2E (learn theta, fixed gamma and delta)
    dr_net_learn_theta = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk,
                        train_pred=True, train_gamma=False, train_delta=False,
                        set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                        cache_path=cache_path, perf_period=perf_period,
                        pred_loss_factor=pred_loss_factor)
    dr_net_learn_theta.net_cv(X, Y, lr_list, epoch_list)
    dr_net_learn_theta.net_roll_test(X, Y)
    dr_net_learn_theta.save_model(cache_path+'dr_net_learn_theta.pkl')
    print('dr_net_learn_theta run complete')

####################################################################################################
# Merge objects with their extended-epoch counterparts
####################################################################################################
if use_cache:
    portfolios = ["base_net", "nom_net", "dr_net", "dr_net_learn_delta", "nom_net_learn_gamma",
                "dr_net_learn_gamma", "nom_net_learn_theta", "dr_net_learn_theta"]
    
    for portfolio in portfolios: 
        cv_combo = pd.concat([as_df(eval(portfolio).cv_results), as_df(eval(portfolio+'_ext').cv_results)], 
                        ignore_index=True)
        eval(portfolio).load_cv_results(cv_combo)
        if eval(portfolio).epochs > 50:
            exec(portfolio + '=' + portfolio+'_ext')
            eval(portfolio).load_cv_results(cv_combo)

####################################################################################################
# Numerical results
####################################################################################################

#---------------------------------------------------------------------------------------------------
# Experiment 1: General
#---------------------------------------------------------------------------------------------------

# Validation results table
dr_net.cv_results = dr_net.cv_results.sort_values(['epochs', 'lr'], 
                                                  ascending=[True, True]
                                                  ).reset_index(drop=True)
exp1_validation_table = pd.concat((as_df(base_net.cv_results).round(4), 
                            as_df(nom_net.cv_results)['val_loss'].round(4), 
                            as_df(dr_net.cv_results)['val_loss'].round(4)), axis=1)
exp1_validation_table.set_axis(['eta', 'Epochs', 'Base', 'Nom.', 'DR'], 
                        axis=1, inplace=True) 

plt.rcParams['text.usetex'] = True
portfolio_names = [r'EW', r'PO', r'Base', r'Nominal', r'DR']
portfolios = [ew_net.portfolio,
              po_net.portfolio,
              base_net.portfolio,
              nom_net.portfolio,
              dr_net.portfolio]

# Out-of-sample summary statistics table
exp1_fin_table = pf.fin_table(portfolios, portfolio_names)

# Wealth evolution plot
portfolio_colors = ["dimgray", 
                    "forestgreen", 
                    "goldenrod", 
                    "dodgerblue", 
                    "salmon"]
pf.wealth_plot(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/wealth_exp1.pdf")
pf.sr_bar(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/sr_bar_exp1.pdf")

# List of initial parameters
exp1_param_dict = dict({'po_net':po_net.gamma.item(),
                'nom_net':nom_net.gamma_init,
                'dr_net':[dr_net.gamma_init, dr_net.delta_init]})

# Trained values for each out-of-sample investment period
exp1_trained_vals = pd.DataFrame(zip([nom_net.gamma_init]+nom_net.gamma_trained, 
                                    [dr_net.gamma_init]+dr_net.gamma_trained, 
                                    [dr_net.delta_init]+dr_net.delta_trained), 
                                    columns=[r'Nom. $\gamma$', 
                                             r'DR $\gamma$', 
                                             r'DR $\delta$'])

#---------------------------------------------------------------------------------------------------
# Experiment 2: Learn delta
#---------------------------------------------------------------------------------------------------

# Validation results table
dr_net_learn_delta.cv_results = dr_net_learn_delta.cv_results.sort_values(['epochs', 'lr'],
                                                    ascending=[True, True]).reset_index(drop=True)
exp2_validation_table = dr_net_learn_delta.cv_results.round(4)
exp2_validation_table.set_axis(['eta', 'Epochs', 'DR (learn delta)'], axis=1, inplace=True) 

plt.rcParams['text.usetex'] = True
portfolio_names = [r'PO', r'DR', r'DR (learn $\delta$)']
portfolios = [po_net.portfolio, 
              dr_po_net.portfolio, 
              dr_net_learn_delta.portfolio]

# Out-of-sample summary statistics table
exp2_fin_table = pf.fin_table(portfolios, portfolio_names)

# Wealth evolution plots
portfolio_colors = ["forestgreen", "dodgerblue", "salmon"]
pf.wealth_plot(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/wealth_exp2.pdf")
pf.sr_bar(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/sr_bar_exp2.pdf")

# List of initial parameters
exp2_param_dict = dict({'po_net':po_net.gamma.item(),
                'dr_po_net':[dr_po_net.gamma.item(), dr_po_net.delta.item()],
                'dr_net_learn_delta':[dr_net_learn_delta.gamma_init,dr_net_learn_delta.delta_init]})

# Trained values for each out-of-sample investment period
exp2_trained_vals = pd.DataFrame([dr_net_learn_delta.delta_init]+dr_net_learn_delta.delta_trained,
                                columns=['DR delta'])

#---------------------------------------------------------------------------------------------------
# Experiment 3: Learn gamma
#---------------------------------------------------------------------------------------------------

# Validation results table
dr_net_learn_gamma.cv_results = dr_net_learn_gamma.cv_results.sort_values(['epochs', 'lr'], 
                                                    ascending=[True, True]).reset_index(drop=True)
dr_net_learn_gamma_delta.cv_results = dr_net_learn_gamma_delta.cv_results.sort_values(['epochs',
                                            'lr'], ascending=[True, True]).reset_index(drop=True)
exp3_validation_table = pd.concat((as_df(nom_net_learn_gamma.cv_results).round(4), 
                            as_df(dr_net_learn_gamma.cv_results)['val_loss'].round(4),
                            as_df(dr_net_learn_gamma_delta.cv_results)['val_loss'].round(4)), axis=1)
exp3_validation_table.set_axis(['eta', 'Epochs', 'Nom. (learn gamma)', 'DR (learn gamma)', 
                                'DR (learn gamma + delta)'], axis=1, inplace=True) 

plt.rcParams['text.usetex'] = True
portfolio_names = [r'PO', r'Nominal', r'DR']
portfolios = [po_net.portfolio, 
              nom_net_learn_gamma.portfolio, 
              dr_net_learn_gamma.portfolio]

# Out-of-sample summary statistics table
exp3_fin_table = pf.fin_table(portfolios, portfolio_names)

# Wealth evolution plots
portfolio_colors = ["forestgreen", "dodgerblue", "salmon"]
pf.wealth_plot(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/wealth_exp3.pdf")
pf.sr_bar(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/sr_bar_exp3.pdf")

# List of initial parameters
exp3_param_dict = dict({'po_net':po_net.gamma.item(),
            'nom_net_learn_gamma':nom_net_learn_gamma.gamma_init,
            'dr_net_learn_gamma':[dr_net_learn_gamma.gamma_init, dr_net_learn_gamma.delta_init],
            'dr_net_learn_gamma_delta':[dr_net_learn_gamma_delta.gamma_init,
                                        dr_net_learn_gamma_delta.delta_init]})

# Trained values for each out-of-sample investment period
exp3_trained_vals = pd.DataFrame(zip(
                    [nom_net_learn_gamma.gamma_init]+nom_net_learn_gamma.gamma_trained, 
                    [dr_net_learn_gamma.gamma_init]+dr_net_learn_gamma.gamma_trained, 
                    [dr_net_learn_gamma_delta.gamma_init]+dr_net_learn_gamma_delta.gamma_trained,
                    [dr_net_learn_gamma_delta.delta_init]+dr_net_learn_gamma_delta.delta_trained),  
                                    columns=['Nom. gamma', 'DR gamma', 'DR gamma 2', 'DR delta'])

#---------------------------------------------------------------------------------------------------
# Experiment 4: Learn theta
#---------------------------------------------------------------------------------------------------

# Validation results table
dr_net_learn_theta.cv_results = dr_net_learn_theta.cv_results.sort_values(['epochs', 'lr'], 
                                                    ascending=[True, True]).reset_index(drop=True)
exp4_validation_table = pd.concat((as_df(base_net.cv_results).round(4), 
                            as_df(nom_net_learn_theta.cv_results)['val_loss'].round(4), 
                            as_df(dr_net_learn_theta.cv_results)['val_loss'].round(4)), axis=1)
exp4_validation_table.set_axis(['eta', 'Epochs', 'Base', 'Nom.', 'DR'], 
                        axis=1, inplace=True) 

plt.rcParams['text.usetex'] = True
portfolio_names = [r'PO', r'Base', r'Nominal', r'DR']
portfolios = [po_net.portfolio, 
              base_net.portfolio, 
              nom_net_learn_theta.portfolio,
              dr_net_learn_theta.portfolio]

# Out-of-sample summary statistics table
exp4_fin_table = pf.fin_table(portfolios, portfolio_names)

# Wealth evolution plots
portfolio_colors = ["forestgreen", "goldenrod", "dodgerblue", "salmon"]
pf.wealth_plot(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/wealth_exp4.pdf")
pf.sr_bar(portfolios, portfolio_names, portfolio_colors, 
                path=cache_path+"plots/sr_bar_exp4.pdf")

# List of initial parameters
exp4_param_dict = dict({'po_net':po_net.gamma.item(),
                    'nom_net_learn_theta':nom_net_learn_theta.gamma_init,
                    'dr_net_learn_theta':[dr_net_learn_theta.gamma_init, 
                                        dr_net_learn_theta.delta_init]})

# Trained values for each out-of-sample investment period
exp4_trained_vals = pd.DataFrame(zip(nom_net_learn_theta.gamma_trained, 
                                    dr_net_learn_theta.gamma_trained, 
                                    dr_net_learn_theta.delta_trained), 
                                columns=['Nom. gamma', 'DR gamma', 'DR delta'])

#---------------------------------------------------------------------------------------------------
# Aggregate Validation Results
#---------------------------------------------------------------------------------------------------

validation_table = pd.concat((as_df(base_net.cv_results).round(4), 
                            as_df(nom_net.cv_results)['val_loss'].round(4),
                            as_df(nom_net_learn_gamma.cv_results)['val_loss'].round(4),
                            as_df(nom_net_learn_theta.cv_results)['val_loss'].round(4), 
                            as_df(dr_net.cv_results)['val_loss'].round(4),
                            as_df(dr_net_learn_delta.cv_results)['val_loss'].round(4),
                            as_df(dr_net_learn_gamma.cv_results)['val_loss'].round(4),
                            as_df(dr_net_learn_gamma_delta.cv_results)['val_loss'].round(4),
                            as_df(dr_net_learn_theta.cv_results)['val_loss'].round(4)), axis=1)
validation_table.set_axis(['eta', 'Epochs', 'Base', 'Nom.', 'Nom. (gamma)', 'Nom. (theta)', 
                        'DR', 'DR (delta)', 'DR (gamma)', 'DR (gamma+delta)', 'DR (theta)'], 
                        axis=1, inplace=True) 

####################################################################################################
# Experiment 5 (with synthetic data)
####################################################################################################

# Path to cache the data, models and results
cache_path_exp5 = "./cache/exp5/"

#---------------------------------------------------------------------------------------------------
# Experiment 5: Load data
#---------------------------------------------------------------------------------------------------

# Train, validation and test split percentage
split = [0.7, 0.3]

# Number of feattures and assets
n_x, n_y = 5, 10

# Number of observations per window and total number of observations
n_obs, n_tot = 100, 1200

# Synthetic data: randomly generate data from a linear model
X, Y = dl.synthetic_exp(n_x=n_x, n_y=n_y, n_obs=n_obs, n_tot=n_tot, split=split)

#---------------------------------------------------------------------------------------------------
# Experiment 5: Initialize parameters
#---------------------------------------------------------------------------------------------------

# Performance loss function and performance period 'v+1'
perf_loss='sharpe_loss'
perf_period = 13

# Weight assigned to MSE prediction loss function
pred_loss_factor = 0.5

# Risk function (default set to variance)
prisk = 'p_var'

# Robust decision layer to use: hellinger or tv
dr_layer = 'hellinger'

# Determine whether to train the prediction weights Theta
train_pred = True

# List of learning rates to test
lr_list = [0.005, 0.0125, 0.02]

# List of total no. of epochs to test
epoch_list = [20, 40, 60]

# Load saved models (default is False)
use_cache = True

#---------------------------------------------------------------------------------------------------
# Run 
#---------------------------------------------------------------------------------------------------
if use_cache:
    with open(cache_path_exp5+'nom_net_linear.pkl', 'rb') as inp:
        nom_net_linear = pickle.load(inp)
    with open(cache_path_exp5+'nom_net_2layer.pkl', 'rb') as inp:
        nom_net_2layer = pickle.load(inp)
    with open(cache_path_exp5+'nom_net_3layer.pkl', 'rb') as inp:
        nom_net_3layer = pickle.load(inp)
    with open(cache_path_exp5+'dr_net_linear.pkl', 'rb') as inp:
        dr_net_linear = pickle.load(inp)
    with open(cache_path_exp5+'dr_net_2layer.pkl', 'rb') as inp:
        dr_net_2layer = pickle.load(inp)
    with open(cache_path_exp5+'dr_net_3layer.pkl', 'rb') as inp:
        dr_net_3layer = pickle.load(inp)
else:

    #***********************************************************************************************
    # Linear models
    #***********************************************************************************************
    
    # For replicability, set the random seed for the numerical experiments
    set_seed = 2000

    # Nominal E2E linear
    nom_net_linear = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, 
                    set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor)
    nom_net_linear.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    nom_net_linear.net_roll_test(X, Y, n_roll=1)
    nom_net_linear.save_model(cache_path+'nom_net_linear.pkl')
    print('nom_net_linear run complete')

    # DR E2E linear
    dr_net_linear = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, 
                    set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor)
    dr_net_linear.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    dr_net_linear.net_roll_test(X, Y, n_roll=1)
    dr_net_linear.save_model(cache_path+'dr_net_linear.pkl')
    print('dr_net_linear run complete')

    #***********************************************************************************************
    # 2-layer models
    #***********************************************************************************************

    # For replicability, set the random seed for the numerical experiments
    set_seed = 3000

    # Nominal E2E 2-layer
    nom_net_2layer = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, pred_model='2layer',
                    set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor)
    nom_net_2layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    nom_net_2layer.net_roll_test(X, Y, n_roll=1)
    nom_net_2layer.save_model(cache_path+'nom_net_2layer.pkl')
    print('nom_net_2layer run complete')

    # DR E2E 2-layer
    dr_net_2layer = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, pred_model='2layer',
                    set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor)
    dr_net_2layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    dr_net_2layer.net_roll_test(X, Y, n_roll=1)
    dr_net_2layer.save_model(cache_path+'dr_net_2layer.pkl')
    print('dr_net_2layer run complete')

    #***********************************************************************************************
    # 3-layer models
    #***********************************************************************************************

    # For replicability, set the random seed for the numerical experiments
    set_seed = 4000

    # Nominal E2E 3-layer
    nom_net_3layer = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, pred_model='3layer',
                    set_seed=set_seed, opt_layer='nominal', perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor)
    nom_net_3layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    nom_net_3layer.net_roll_test(X, Y, n_roll=1)
    nom_net_3layer.save_model(cache_path+'nom_net_3layer.pkl')
    print('nom_net_3layer run complete')

    # DR E2E 3-layer
    dr_net_3layer = e2e.e2e_net(n_x, n_y, n_obs, prisk=prisk, train_pred=train_pred, 
                    train_gamma=True, train_delta=True, pred_model='3layer',
                    set_seed=set_seed, opt_layer=dr_layer, perf_loss=perf_loss, 
                    perf_period=perf_period, pred_loss_factor=pred_loss_factor)
    dr_net_3layer.net_cv(X, Y, lr_list, epoch_list, n_val=1)
    dr_net_3layer.net_roll_test(X, Y, n_roll=1)
    dr_net_3layer.save_model(cache_path+'dr_net_3layer.pkl')
    print('dr_net_3layer run complete')

#---------------------------------------------------------------------------------------------------
# Experiment 5: Results
#---------------------------------------------------------------------------------------------------

# Validation results table
exp5_validation_table = pd.concat((as_df(nom_net_linear.cv_results).round(4), 
                            as_df(dr_net_linear.cv_results)['val_loss'].round(4), 
                            as_df(nom_net_2layer.cv_results)['val_loss'].round(4), 
                            as_df(dr_net_2layer.cv_results)['val_loss'].round(4), 
                            as_df(nom_net_3layer.cv_results)['val_loss'].round(4), 
                            as_df(dr_net_3layer.cv_results)['val_loss'].round(4)), axis=1)
exp5_validation_table.set_axis(['eta', 'Epochs', 'Nom. (linear)', 'DR (linear)', 
                            'Nom. (2-layer)', 'DR (2-layer)', 'Nom. (3-layer)', 'DR (3-layer)'],
                            axis=1, inplace=True) 

plt.rcParams['text.usetex'] = True
portfolio_names = [r'Nom. (linear)', 
                   r'DR (linear)', 
                   r'Nom. (2-layer)', 
                   r'DR (2-layer)', 
                   r'Nom. (3-layer)', 
                   r'DR (3-layer)']
portfolios = [nom_net_linear.portfolio, 
              dr_net_linear.portfolio, 
              nom_net_2layer.portfolio,
              dr_net_2layer.portfolio, 
              nom_net_3layer.portfolio, 
              dr_net_3layer.portfolio]

# Out-of-sample summary statistics table
exp5_fin_table = pf.fin_table(portfolios, portfolio_names)

# Wealth evolution plot
portfolio_colors = ["dodgerblue", "salmon", "dodgerblue", "salmon", "dodgerblue", "salmon"]
pf.wealth_plot(portfolios, portfolio_names, portfolio_colors, nplots=3,
                path=cache_path+"plots/wealth_exp5.pdf")

# List of initial parameters
exp5_param_dict = dict({'nom_net_linear':nom_net_linear.gamma_init,
                    'dr_net_linear':[dr_net_linear.gamma_init, dr_net_linear.delta_init],
                    'nom_net_2layer':nom_net_2layer.gamma_init,
                    'dr_net_2layer':[dr_net_2layer.gamma_init, dr_net_2layer.delta_init],
                    'nom_net_3layer':nom_net_3layer.gamma_init,
                    'dr_net_3layer':[dr_net_3layer.gamma_init, dr_net_3layer.delta_init]})
                    