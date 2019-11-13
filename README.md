A feed-forward neural network based software for treatment effect and propensity score estimation (in case of observational data). It is supporting both regression and classification treatment variable case.

### Installation

First, clone the software repository, then install the dependencies and finally the software itself, by typing the following commands.

```sh
cd install/path
git clone https://github.com/PopovicMilica/causal_nets.git
cd causal_nets
pip install -r requirements.txt
pip install .
cd ..
```
Test installation by importing the package:
```Python
import causal_nets
```

### Example

The below code is a short example showcasing the usage of causal_nets software.

```Python
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from causal_nets import causal_net_estimate
import matplotlib.pyplot as plt

# Setting the seeds
np.random.seed(3)
tf.compat.v1.set_random_seed(12)

# Generating the fake data
N = 10000
X = np.random.uniform(low=0, high=1, 
                      size=[N, 10])
mu0_real = 0.012*X[:,3] - 0.75*X[:,5]*X[:,7] - 0.9*X[:, 4] - np.mean(X, axis = 1)
tau_real = X[:,2] + 0.04*X[:,9] - 0.35*np.log(X[:, 3]) #+ 0.5*np.sum(X[:,3:],axis = 1) # 
prob_of_T = 0.5
T = np.random.binomial(size=N, n=1, p=prob_of_T)
normal_errors = np.random.normal(size=[N,], loc=0.0, scale=1.0)
Y = mu0_real + tau_real*T + normal_errors

# Creating training and validation dataset
X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(
    X, T, Y, test_size = 0.3, random_state=88)

# Getting causal estimates 
tau_pred, mu0_pred, prob_t_pred = causal_net_estimate(
    0, 1, 2, [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], X,
    [30, 20, 15, 10, 5], dropout_rates=None, batch_size=None, alpha_l1=0.,
    alpha_l2=0., optimizer='Adam', learning_rate=0.0009,
    max_epochs_without_change=30, max_nepochs=10000,
    distribution='LinearRegression', estimate_ps=False, plot_coeffs = True)

# Plotting estimated coefficient vs real coefficients
def plotting_treatment_coefficients(tau_pred, mu0_pred):
    
    # Plotting treatment effect coefficient distributions
    plt.figure(figsize=(12, 10))
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.hist(tau_pred, alpha=0.6, label='tau_pred', normed=True)
    plt.hist(tau_real, label='tau_real', histtype=u'step',
             normed=True, linewidth=2.5)
    plt.legend(loc='upper right')
    plt.title('CATE(Conditional average treatment effect)')
    plt.xlabel('tau', fontsize=14)
    plt.ylabel('Density')
    
    plt.subplot(2, 2, 2)
    plt.plot(tau_pred,tau_real,'k.',markersize = 4)
    plt.plot(tau_real,tau_real,'r-',lw=1)
    plt.xlabel('tau_trained_',fontsize = 14)
    plt.ylabel('tau_real_',fontsize = 14)
    plt.xlim(np.min(tau_real),np.max(tau_real))
    plt.ylim(np.min(tau_real),np.max(tau_real))
    plt.title('Tau coefficients comparison')
    
    plt.subplot(2, 2, 3)
    plt.hist(mu0_pred, alpha=0.7, label=r'$\mu_0$_pred', normed=True)
    plt.hist(mu0_real,label=r'$\mu_0$_real', histtype=u'step',
             normed=True, linewidth=2.5)
    plt.legend(loc='upper right')
    plt.title(r'$\mu_0(x)$')
    plt.xlabel('mu0', fontsize=14)
    plt.ylabel('Density')
    
    plt.subplot(2, 2, 4)
    plt.plot(mu0_pred, mu0_real,'k.',markersize = 4)
    plt.plot(mu0_real, mu0_real,'r-',lw=1)
    plt.xlabel('mu0_pred',fontsize = 14)
    plt.ylabel('mu0_real',fontsize = 14)
    plt.xlim(np.min(mu0_real),np.max(mu0_real))
    plt.ylim(np.min(mu0_real),np.max(mu0_real))
    plt.title(r'$\mu_0(x)$ coefficients comparison')
    plt.tight_layout()
    plt.show()
    
plotting_treatment_coefficients(tau_pred, mu0_pred)
```
### Explanation of the parameters of the main function causal_net_estmates()
```
def causal_net_estimate(ind_X, ind_T, ind_Y, training_data, validation_data,
                        test_data,  hidden_layer_sizes, dropout_rates=None,
                        batch_size=None, alpha_l1=0., alpha_l2=0.,
                        optimizer='Adam', learning_rate=0.0009,
                        max_epochs_without_change=30, max_nepochs=5000,
                        distribution='LinearRegression', estimate_ps=False,
                        hidden_layer_sizes_t=None, dropout_rates_t=None,
                        batch_size_t=None, alpha_l1_t=0., alpha_l2_t=0.,
                        optimizer_t='Adam', learning_rate_t=0.0009,
                        max_epochs_without_change_t=30, max_nepochs_t=5000,
                        plot_coeffs=True)
```

    Parameters  
    ----------  
    ind_X: {0, 1, 2}  
        Features array index in data list.  
    ind_T: {0, 1, 2}  
        Treatment array index in data list.  
    ind_Y: {0, 1, 2}  
        Target array index in data list.
    training_data: list of arrays
        Data on which the training of the Neural Network will be
        performed. It is comprised as a list of arrays, in the
        following manner:
        [X_train, T_train, Y_train] or [T_train, X_train, Y_train] or
        any other ordering of the three arrays, just be carful when you
        specify the above indices as the mapping to be correct. Also,
        once you choose your ordering, be consistant to preserve
        ordering in all the data arrays.
        Here, `X_train` is an array of input features, `T_train` is
        the treatment array, and `Y_train` is the target array.
    validation_data: list of arrays
        Data on which the validation of the Neural Network will be
        performed. It has to be composed in the same manner as the
        training data.
    test_data: array like
        A feature array on which we want to perform estimation.
    hidden_layer_sizes: list of ints
        `hidden_layer_sizes` is a list that defines a size and width of
        the neural network that estimates causal coefficients. Length of
        the list defines the number of hidden layers. Entries of the
        list define the number of hidden units in each hidden layer.
        No default value, it needs to be provided.
        E.g. hidden_layer_sizes = [60, 30]
    dropout_rates: list of floats or None, optional
        If it's a list than the values of the list represent dropout
        rate for each layer of the neural network that estimates causal
        coefficients. Each entry of the list has to be between 0 and 1.
        Also, list has to be of same length as the list
        'hidden_layer_sizes'. If is set to None, than dropout is not
        applied. Default value is None.
    batch_size: int, optional
        Batch size for the neural network that estimates causal
        coefficients. Default value is None. If batch_size is None,
        than batch size is equal to length of the training dataset for
        training datasets smaller than 50000 rows and set to 1024 for
        larger datasets. Otherwise, it is equal to the value provided.
    alpha_l1: float, optional
        Lasso(L1) regularization factor for the neural network that
        estimates causal coefficients. Default value is 0.
    alpha_l2: float, optional
        Ridge(L2) regularization factor for the neural network that
        estimates causal coefficients. Default value is 0.
    optimizer: str, optional
        Which optimizer to use for the neural network that estimates
        causal coefficients. See page: https://keras.io/optimizers/ for
        optimizers options. Default: 'Adam'.
    learning_rate: scalar, optional
        Learning rate for the neural network that estimates
        causal coefficients. Default value is 0.0009.
    max_epochs_without_change: int, optional
        Number of epochs with no improvement on the validation loss to
        wait before stopping the training for the neural network that
        estimates causal coefficients. Default value is 30.
    max_nepochs: int, optional
        Maximum number of epochs for which neural network, that
        estimates causal coefficients, will be trained.
        Default value is 5000.
    distribution: {'Sigmoid', 'LinearRegression'}, optional
        If the target variable for causal coefficients is a binary
        categorical, then use distribution='Sigmoid'. Otherwise, if the
        target variable is continuous then distribution='LinearRegression'.
        Default: 'LinearRegression'
    estimate_ps:False, optional
    hidden_layer_sizes_t=None:, optional
        `hidden_layer_sizes_t` is a list that defines a size and width
        of the neural network that estimates propensity scores. Length
        of the list defines the number of hidden layers. Entries of the
        list define the number of hidden units in each hidden layer.
        Default value is None, but if 'estimate_ps' is set to True,
        than the values for this argument needs to be provided.
        E.g. hidden_layer_sizes_t = [60, 30]
    dropout_rates_t: list of floats or None, optional
        If it's a list than the values of the list represent dropout
        rate for each layer of the neural network that estimates
        propensity scores. Each entry of the list has to be between 0
        and 1. Also, list has to be of same length as the list
        'hidden_layer_sizes_t'. If is set to None, than dropout is not
        applied.
        Default value is None.
    batch_size_t: int, optional
        Batch size for the neural network that estimates propensity
        scores. Default value is None. If batch_size is None, than
        batch size is equal to the length of the training dataset for
        training datasets smaller than 50000 rows and set to 1024 for
        larger datasets. Otherwise, it is equal to the value provided.
    alpha_l1_t: float, optional
        Lasso(L1) regularization factor for the neural network that
        estimates propensity scores. Default value is 0.
    alpha_l2_t: float, optional
        Ridge(L2) regularization factor for the neural network that
        estimates propensity scores. Default value is 0.
    optimizer_t: str, optional
        Which optimizer to use for the neural network that estimates
        propensity scores. See page: https://keras.io/optimizers/ for
        optimizers options. Default: 'Adam'.
    learning_rate_t: scalar, optional
        Learning rate for the neural network that estimates propensity
        scores. Default value is 0.0009.
    max_epochs_without_change_t: int, optional
        Number of epochs with no improvement on the validation loss to
        wait before stopping the training for the neural network that
        estimates propensity scores. Default value is 30.
    max_nepochs_t int, optional
        Maximum number of epochs for which neural network, that
        estimates propensity scores, will be trained.
        Default value is 5000.
    plot_coeffs: bool, optional
        Should estimated values be plotted or not. Default: True

    Returns
    -------
    tau_pred: ndarray
        Estimated conditional average treatment effect.
    mu0_pred: ndarray
        Estimated intercept.
    prob_of_t_pred:: ndarray
        Estimated propensity scores

### References

Farrell, M.H., Liang, T. and Misra, S., 2018:
'Deep neural networks for estimation and inference: Application to causal effects and other semiparametric estimands',
[<a href="https://arxiv.org/pdf/1809.09953.pdf">arxiv</a>]
