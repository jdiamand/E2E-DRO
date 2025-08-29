# PlotFunctions Module
#
####################################################################################################
## Import libraries
####################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import gmean

# Matplotlib parameters
plt.close("all")
plt.rcParams["font.family"] ="serif"
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['text.usetex'] = True

####################################################################################################
# Ploting functions
####################################################################################################

#---------------------------------------------------------------------------------------------------
# wealth_plot function
#---------------------------------------------------------------------------------------------------
def wealth_plot(portfolio_list, names, colors, nplots=1, path=None):
    """Plot of the portfolio wealth evolution over time (also known as the 'Total Return Index')

    Inputs
    portfolio_list: list of portfolio objects corresponding to the backtest of each model
    names: list of strings with the portfolio names that shall appear in the plot legend
    colors: list of strings with matplotlib color names to be used for each portfolio
    nplots: Number of subplots into which to distribute the results
    path: Path to which to save the image in pdf format. If 'None', then the image is not saved

    Output
    Wealth evolution figure
    """
    n = len(portfolio_list)
    # Normalize portfolios that stored numpy fallback
    tri_series = []
    for i in range(n):
        rets = portfolio_list[i].rets
        if isinstance(rets, np.ndarray):
            # columns: [index, rets, tri]
            tri = pd.Series(rets[:, 2], name=names[i])
            tri.index = pd.RangeIndex(start=0, stop=len(tri))
        else:
            tri = rets.tri.rename(names[i])
        tri_series.append(tri * 100)
    plot_df = pd.concat(tri_series, axis=1)
    s = pd.DataFrame([100*np.ones(n)], columns=names)
    if isinstance(plot_df.index, pd.DatetimeIndex):
        s.index = [plot_df.index[0] - pd.Timedelta(days=7)]
    else:
        s.index = [plot_df.index[0] - 1]
    plot_df = pd.concat([s, plot_df])

    if nplots == 1:
        fig, ax = plt.subplots(figsize=(6,4))
        for i in range(n):
            ax.plot(plot_df[names[i]], color=colors[i])
        ax.legend(names, ncol=n, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                handlelength=1)
        # ax.legend(names, fontsize=14)
        ax.grid(b="on",linestyle=":",linewidth=0.8)
        ax.tick_params(axis='x', labelrotation = 30)
        plt.ylabel("Total wealth", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(max([6, nplots*4]),4), ncols=nplots)
        for i in range(n):
            j = int(nplots/n * i)
            ax[j].plot(plot_df[names[i]], color=colors[i])
            if j == 0:
                ax[j].set_ylabel("Total wealth", fontsize=14)
            ax[j].tick_params(axis='both', which='major', labelsize=14)

        for j in range(nplots):
            i = int(j * n / nplots)
            k = int((j+1) * n / nplots)
            ax[j].legend(names[i:k], ncol=int(n / nplots), fontsize=12, loc='upper center', 
                        bbox_to_anchor=(0.5, -0.15), handlelength=1)
            # ax[j].legend(names[i:k], fontsize=14)
            ax[j].grid(visible="on",linestyle=":",linewidth=0.8)
            ax[j].tick_params(axis='x', labelrotation = 30)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path[0:-3]+'ps', bbox_inches='tight', format='ps')

#---------------------------------------------------------------------------------------------------
# sr_plot function
#---------------------------------------------------------------------------------------------------
def sr_plot(portfolio_list, names, colors, path=None):
    """Plot of the Sharpe ratio calculated over a rolling 2-year period
    
    Inputs
    portfolio_list: list of portfolio objects corresponding to the backtest of each model
    names: list of strings with the portfolio names that shall appear in the plot legend
    colors: list of strings with matplotlib color names to be used for each portfolio

    Output
    SR evolution figure
    """
    time_period = 104
    series = []
    for i in range(len(portfolio_list)):
        rets = portfolio_list[i].rets
        if isinstance(rets, np.ndarray):
            s = pd.Series(rets[:, 1], name=names[i])
            s.index = pd.RangeIndex(start=0, stop=len(s))
        else:
            s = rets.rets.rename(names[i])
        series.append(s)
    df = pd.concat(series, axis=1)
    mean_df = ((df+1).rolling(time_period).apply(gmean))**52 - 1
    mean_df.dropna(inplace=True)
    std_df = df.rolling(time_period).std()
    std_df.dropna(inplace=True)
    plot_df = mean_df / (std_df * np.sqrt(52))

    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(len(portfolio_list)):
        ax.plot(plot_df[names[i]], color=colors[i])

    ax.legend(names, ncol=3, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.tick_params(axis='x', labelrotation = 30)
    plt.ylabel("2-yr SR", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path[0:-3]+'eps', bbox_inches='tight', format='eps')

#---------------------------------------------------------------------------------------------------
# sr_plot function
#---------------------------------------------------------------------------------------------------
def sr_bar(portfolio_list, names, colors, path=None):
    """Plot of the Sharpe ratio calculated over a rolling 2-year period
    
    Inputs
    portfolio_list: list of portfolio objects corresponding to the backtest of each model
    names: list of strings with the portfolio names that shall appear in the plot legend
    colors: list of strings with matplotlib color names to be used for each portfolio

    Output
    SR evolution figure
    """
    n = len(portfolio_list)
    series = []
    for i in range(n):
        rets = portfolio_list[i].rets
        if isinstance(rets, np.ndarray):
            s = pd.Series(rets[:, 1], name=names[i])
            s.index = pd.RangeIndex(start=0, stop=len(s))
        else:
            s = rets.rets.rename(names[i])
        series.append(s)
    df = pd.concat(series, axis=1)
    
    mean_df = df.expanding(min_periods=1).mean().groupby([df.index.year]).tail(1)
    std_df  = df.expanding(min_periods=1).std().groupby([df.index.year]).tail(1)
    plot_df = mean_df / std_df * np.sqrt(52)

    x = np.arange(plot_df.shape[0])
    w = 1/n
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(n):
        ax.bar(x - 0.5 + i/n, plot_df[names[i]], w, color=colors[i])

    ax.legend(names, ncol=n, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
            handlelength=1)
    ax.grid(b="on",linestyle=":",linewidth=0.8)
    ax.set_xticks(x, plot_df.index.year.to_list())
    
    ax.set_xticks(np.arange(-.6, plot_df.shape[0], 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=1)
    ax.grid(which='major', color='w', linestyle='-', linewidth=0)

    plt.ylabel("Sharpe ratio", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path[0:-3]+'eps', bbox_inches='tight', format='eps')

#---------------------------------------------------------------------------------------------------
# learn_plot function
#---------------------------------------------------------------------------------------------------
def learn_plot(trained_vals, colors, marker, delta_mark, path=None):
    """Plot of the Sharpe ratio calculated over a rolling 2-year period
    
    Inputs
    trained_vals: pd.Dataframe of learned parameters 
    colors: list of strings with matplotlib color names to be used for each portfolio

    Output
    Plot of learned parameters (gamma as bar, delta as line)
    """

    t, n = trained_vals.shape
    x = np.linspace(0, t-1, num=t)

    fig, ax = plt.subplots(figsize=(6,4))
    ax2 = ax.twinx()
    for i in range(n):
        if i < delta_mark:
            ax.stem(x+i/5, trained_vals.iloc[:,i], colors[i], markerfmt=marker[i],
            bottom=trained_vals.iloc[0,i])
        else:
            ax2.stem(x+i/5, trained_vals.iloc[:,i], colors[i], markerfmt=marker[i],
            bottom=trained_vals.iloc[0,i])

    ax.legend(trained_vals.columns, ncol=n, fontsize=12, loc='upper center', 
            bbox_to_anchor=(0.5, -0.15), handlelength=1)
    ax.grid(b="on",linestyle=":",linewidth=0.8)

    ax.set_xlabel(r'Training period', fontsize=14)
    ax.set_ylabel(r'$\gamma$', fontsize=14)
    if i < delta_mark:
        ax2.set_ylabel(r'$\delta$', fontsize=14)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if path is not None:
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path[0:-3]+'eps', bbox_inches='tight', format='eps')
        
####################################################################################################
# Other results
####################################################################################################
#---------------------------------------------------------------------------------------------------
# fin_table
#---------------------------------------------------------------------------------------------------
def fin_table(portfolios:list, names:list) -> pd.DataFrame:
    """Compute portfolio performance statistics and summarize them as a table
    
    Inputs
    List of backtest-type objects
    
    Outputs
    Table of results    
    """
    
    rets =[]
    vols = []
    SRs = []
    invHidxs = []
    
    for portfolio in portfolios:
        rets_obj = portfolio.rets
        if isinstance(rets_obj, np.ndarray):
            tri = rets_obj[:, 2]
            n = tri.shape[0]
            ret = (tri[-1] ** (1/n))**52 - 1
        else:
            ret = (rets_obj.tri.iloc[-1] ** (1/rets_obj.tri.shape[0]))**52 - 1
        # Ensure volatility is scalar
        vol_arr = np.asarray(portfolio.vol)
        vol_scalar = float(np.mean(vol_arr)) * np.sqrt(52)
        SR = float(ret) / vol_scalar if vol_scalar != 0 else float('nan')
        # Numpy-only inverse HHI across rows to avoid pandas issues
        weights_arr = np.asarray(portfolio.weights, dtype=float)
        hhi_rows = np.sum(np.square(weights_arr), axis=1)
        invHidx_val = float(np.round(np.mean(1.0 / np.maximum(hhi_rows, 1e-12)), 2))
        rets.append(round(float(ret)*100, ndigits=1))
        vols.append(round(vol_scalar*100, ndigits=1))
        SRs.append(round(float(SR), ndigits=2))
        invHidxs.append(invHidx_val)
    # Build via numeric matrix, then assign safe column names
    k = min(len(rets), len(vols), len(SRs), len(invHidxs), len(names))
    rets, vols, SRs, invHidxs = rets[:k], vols[:k], SRs[:k], invHidxs[:k]
    # Coerce names to safe python strings; replace non-strings/arrays
    safe_names = []
    for i in range(k):
        n = names[i]
        try:
            if isinstance(n, (list, np.ndarray)):
                safe_names.append(f"Portfolio {i+1}")
            else:
                safe_names.append(str(n))
        except Exception:
            safe_names.append(f"Portfolio {i+1}")

    # Build via pure Python lists, then transpose so metrics are rows
    metrics = ['Return (%)', 'Volatility (%)', 'Sharpe ratio', 'Avg. inv. HHI']
    rows = []
    for idx in range(k):
        rows.append([float(rets[idx]), float(vols[idx]), float(SRs[idx]), float(invHidxs[idx])])
    try:
        table = pd.DataFrame(rows, columns=metrics)
        try:
            table.index = safe_names
        except Exception:
            table.index = [f"Portfolio {i+1}" for i in range(k)]
        table = table.T
        return table
    except Exception as e:
        print("🔍 DEBUG: Using numpy fallback for fin_table due to:", e)
        # Return a pandas-free structure to avoid crashing the run
        return {"metrics": metrics, "names": safe_names, "rows": rows}
