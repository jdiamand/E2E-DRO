# Financial performance loss functions for E2E learning framework
#
####################################################################################################
## Import libraries
####################################################################################################
import torch

####################################################################################################
# Performance loss functions
####################################################################################################
def single_period_loss(z_star, y_perf):
    """Loss function based on the out-of-sample portfolio return

    Compute the out-of-sample portfolio return for portfolio z_t over the next time step. The
    loss is aggregated for all z_t in Z_star and averaged over the number of observations.

    Inputs
    z_star: Optimal solution. (n_y x 1) tensor of optimal decisions.
    y_perf: Realizations. (perf_period x n_y) tensor of realized values.

    Output
    loss: realized return at time 't' 
    """
    # Ensure inputs maintain gradient flow
    if not z_star.requires_grad:
        z_star = z_star.detach().requires_grad_(True)
    
    # Compute loss (negative return for minimization)
    loss = -y_perf[0] @ z_star 
    return loss

def single_period_over_var_loss(z_star, y_perf):
    """Loss function based on the out-of-sample portfolio return over volatility

    Compute the out-of-sample portfolio return for portfolio z_star over the next time step. Divide
    by the realized volatility over the performance period ('perf_period')

    Inputs
    z_star: Optimal solution. (n_y x 1) tensor of optimal decisions.
    y_perf: Realizations. (perf_period x n_y) tensor of realized values.

    Output
    loss: realized return at time 't' over realized volatility from 't' to 't + perf_period'
    """
    # Ensure inputs maintain gradient flow
    if not z_star.requires_grad:
        z_star = z_star.detach().requires_grad_(True)
    
    # Compute portfolio returns
    portfolio_returns = y_perf @ z_star
    
    # Compute standard deviation with numerical stability
    std_return = torch.std(portfolio_returns)
    
    # Add small epsilon to prevent division by zero and maintain numerical stability
    epsilon = 1e-8
    std_return = std_return + epsilon
    
    # Compute loss (negative return over volatility)
    loss = -y_perf[0] @ z_star / std_return
    return loss

def sharpe_loss(z_star, y_perf):
    """Loss function based on the out-of-sample Sharpe ratio

    Compute the out-of-sample Sharpe ratio of the portfolio z_t over the next 12 time steps. The
    loss is aggregated for all z_t in Z_star and averaged over the number of observations. We use a
    simplified version of the Sharpe ratio, SR = realized mean / realized std dev.

    Inputs
    z_star: Optimal solution. (n_y x 1) tensor of optimal decisions.
    y_perf: Realizations. (perf_period x n_y) tensor of realized values.

    Output
    loss: realized average return over realized volatility from 't' to 't + perf_period'
    """
    # Ensure inputs maintain gradient flow
    if not z_star.requires_grad:
        z_star = z_star.detach().requires_grad_(True)
    
    # Compute portfolio returns
    portfolio_returns = y_perf @ z_star
    
    # Compute mean and standard deviation with numerical stability
    mean_return = torch.mean(portfolio_returns)
    std_return = torch.std(portfolio_returns)
    
    # Add small epsilon to prevent division by zero and maintain numerical stability
    epsilon = 1e-8
    std_return = std_return + epsilon
    
    # Compute Sharpe ratio (negative for minimization)
    loss = -mean_return / std_return
    
    return loss