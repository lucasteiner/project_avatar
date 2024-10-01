import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import emcee

def linear_regression_with_plot(x, y, names):
    if len(x) != len(y) or len(x) != len(names):
        raise ValueError("Length of x, y, and names must be the same.")

    # Perform linear regression (least squares fit)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate R^2 value
    r_squared = 1 - np.sum((y - regression_line) ** 2) / np.sum((y - np.mean(y)) ** 2)

    # Plot data points with unique colors and labels
    #colors = plt.cm.get_cmap('tab10', len(x))  # Using a colormap for unique colors

    for i in range(len(x)):
        #plt.scatter(x[i], y[i], color=colors(i), label=names[i])
        plt.scatter(x[i], y[i], label=names[i])
        plt.text(x[i], y[i], names[i], fontsize=9, ha='right', va='bottom')

    # Plot the regression line
    plt.plot(x, regression_line, color='red', label='Regression line')

    # Dynamically place text based on the data range
    #text_x = min(x) + (max(x) - min(x)) * 0.05
    #text_y = max(y) - (max(y) - min(y)) * 0.05
    #plt.text(text_x, text_y, f'Slope: {slope:.2f}\nIntercept: {intercept:.2f}\n$R^2$: {r_squared:.4f}', 
    #         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Linear Regression (Least Squares Fit)')
    #plt.legend()  # Enable legend
    plt.show()

    return slope, intercept, r_squared

def log_prior(theta):
    """Log-prior: uniform distribution over reasonable ranges."""
    slope, intercept, sigma = theta
    if -10.0 < slope < 10.0 and -10.0 < intercept < 10.0 and 0.0 < sigma < 10.0:
        return 0.0  # log(1) is 0
    return -np.inf  # log(0) is -inf

def log_likelihood(theta, x, y):
    """Log-likelihood of the observed data given the model parameters."""
    slope, intercept, sigma = theta
    model = slope * x + intercept
    return -0.5 * np.sum(((y - model) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

def log_posterior(theta, x, y):
    """Log-posterior combines prior and likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y)

def bayesian_regression_with_plot(x, y, names):
    if len(x) != len(y) or len(x) != len(names):
        raise ValueError("Length of x, y, and names must be the same.")

    # Initial guess for parameters: slope, intercept, sigma
    initial = np.array([0.5, 0.5, 1.0])
    n_walkers = 50
    n_dim = len(initial)

    # Set up the sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior, args=(x, y))
    p0 = initial + 1e-4 * np.random.randn(n_walkers, n_dim)

    # Run MCMC
    sampler.run_mcmc(p0, 5000, progress=True)
    samples = sampler.get_chain(discard=100, thin=10, flat=True)

    # Extract parameter estimates
    slope_mcmc, intercept_mcmc, sigma_mcmc = np.mean(samples, axis=0)

    # Plot data points with unique colors and labels
    #colors = plt.cm.get_cmap('tab10', len(x))  # Using a colormap for unique colors
    for i in range(len(x)):
        #plt.scatter(x[i], y[i], color=colors(i), label=names[i])
        plt.scatter(x[i], y[i], label=names[i])
        plt.text(x[i], y[i], names[i], fontsize=9, ha='right', va='bottom')

    # Plot the Bayesian regression line
    regression_line = slope_mcmc * x + intercept_mcmc
    plt.plot(x, regression_line, color='red', label='BRL') # Bayesian Regression line

    # Dynamically place text based on the data range
    #text_x = min(x) + (max(x) - min(x)) * 0.05
    #text_y = max(y) - (max(y) - min(y)) * 0.05
    #plt.text(text_x, text_y, f'Slope: {slope_mcmc:.2f}\nIntercept: {intercept_mcmc:.2f}', 
    #         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Bayesian Linear Regression')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 1))  # Enable legend
    plt.show()

    return slope_mcmc, intercept_mcmc, sigma_mcmc

# Example usage:
#x = np.array([1, 2, 3, 4, 5])
#y = np.array([2, 3, 5, 7, 11])
#names = ["Point A", "Point B", "Point C", "Point D", "Point E"]

#slope, intercept, sigma = bayesian_regression_with_plot(x, y, names)

def log_prior_yerr(theta):
    """Log-prior_yerr: uniform distribution over reasonable ranges."""
    slope, intercept, log_sigma = theta
    if -10.0 < slope < 10.0 and -10.0 < intercept < 10.0 and -5.0 < log_sigma < 1.0:
        return 0.0  # log(1) is 0
    return -np.inf  # log(0) is -inf

def log_likelihood_yerr(theta, x, y, yerr):
    """Log-likelihood_yerr of the observed data given the model parameters and uncertainties."""
    slope, intercept, log_sigma = theta
    sigma = np.exp(log_sigma)  # Convert log_sigma back to sigma
    model = slope * x + intercept
    return -0.5 * np.sum(((y - model) / yerr) ** 2 + np.log(2 * np.pi * (yerr ** 2 + sigma ** 2)))

def log_posterior_yerr(theta, x, y, yerr):
    """Log-posterior_yerr combines prior and likelihood."""
    lp = log_prior_yerr(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_yerr(theta, x, y, yerr)

def bayesian_regression_with_plot_yerr(x, y, yerr, names):
    if len(x) != len(y) or len(x) != len(names) or len(x) != len(yerr):
        raise ValueError("Length of x, y, yerr, and names must be the same.")

    # Initial guess for parameters: slope, intercept, log(sigma)
    initial = np.array([0.5, 0.5, 0.0])
    n_walkers = 50
    n_dim = len(initial)

    # Set up the sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior_yerr, args=(x, y, yerr))
    p0 = initial + 1e-4 * np.random.randn(n_walkers, n_dim)

    # Run MCMC
    sampler.run_mcmc(p0, 5000, progress=True)
    samples = sampler.get_chain(discard=100, thin=10, flat=True)

    # Extract parameter estimates
    slope_mcmc, intercept_mcmc, log_sigma_mcmc = np.mean(samples, axis=0)
    sigma_mcmc = np.exp(log_sigma_mcmc)

    # Plot data points with unique colors and labels
    #colors = plt.cm.get_cmap('tab10', len(x))  # Using a colormap for unique colors
    for i in range(len(x)):
        #plt.errorbar(x[i], y[i], yerr=yerr[i], fmt='o', color=colors(i), label=names[i], capsize=5)
        plt.errorbar(x[i], y[i], yerr=yerr[i], fmt='o', label=names[i], capsize=5)
        plt.text(x[i], y[i], names[i], fontsize=9, ha='right', va='bottom')

    # Plot the Bayesian regression line
    regression_line = slope_mcmc * x + intercept_mcmc
    plt.plot(x, regression_line, color='red', label='BRL')

    # Dynamically place text based on the data range
    #text_x = min(x) + (max(x) - min(x)) * 0.05
    #text_y = max(y) - (max(y) - min(y)) * 0.05
    #plt.text(text_x, text_y, f'Slope: {slope_mcmc:.2f}\nIntercept: {intercept_mcmc:.2f}\nSigma: {sigma_mcmc:.2f}', 
    #         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Bayesian Linear Regression with Uncertainties')
    #plt.legend(loc='center left', bbox_to_anchor=(1, 1))  # Enable legend
    plt.show()

    return slope_mcmc, intercept_mcmc, sigma_mcmc

## Example usage:
#x = np.array([1, 2, 3, 4, 5])
#y = np.array([2, 3, 5, 7, 11])
#yerr = np.array([0.5, 0.2, 0.3, 0.4, 0.1])
#names = ["Point A", "Point B", "Point C", "Point D", "Point E"]
#
#slope, intercept, sigma = bayesian_regression_with_plot_yerr(x, y, yerr, names)