A feed-forward neural network based software for treatment effects and propensity score estimation. Currently, it only supports the case when target variable is continuous.

### Installation

First, navigate to the folder where you want to store the software repository, then clone the software repository. After that install the dependencies, and finally the software itself by typing the following commands:

```sh
cd install/path
git clone https://github.com/PopovicMilica/causal_nets.git
cd causal_nets
pip3 install -r requirements.txt
pip3 install .
cd ..
```
Test installation by importing the package:
```Python
import causal_nets
```

Note: This software supports only Python versions 3.6 and above.
### Example

The below code is a short example showcasing the usage of causal_nets software.

```Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from causal_nets import causal_net_estimate


# Setting the seeds
np.random.seed(3)

# Generating the fake data
N = 10000
X = np.random.uniform(low=0, high=1, size=[N, 10])
mu0_real = 0.012*X[:, 3] - 0.75*X[:, 5]*X[:, 7] - 0.9*X[:, 4] - np.mean(X, axis=1)
tau_real = X[:, 2] + 0.04*X[:, 9] - 0.35*np.log(X[:, 3])
prob_of_T = 0.5
T = np.random.binomial(size=N, n=1, p=prob_of_T)
normal_errors = np.random.normal(size=[N, ], loc=0.0, scale=1.0)
Y = mu0_real + tau_real*T + normal_errors

# Creating training and validation dataset
X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(
    X, T, Y, test_size=0.2, random_state=42)

# Getting causal estimates
tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1, history, history_ps = causal_net_estimate(
    [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid], [X, T, Y], [30, 20, 15, 10, 5],
    dropout_rates=None, batch_size=None, alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
    max_epochs_without_change=30, max_nepochs=10000, seed=None, estimate_ps=False, verbose=True)

# Plotting estimated coefficient vs true coefficients    
plt.figure(figsize=(10, 5))
plt.clf()

plt.subplot(1, 2, 1)
bins = np.linspace(min(min(tau_pred),min(tau_real)),
                    max(max(tau_pred),max(tau_real)), 15)
plt.hist(tau_pred, alpha=0.6, label=r'$\tau~_{pred}$',
         density=True, bins=bins)
plt.hist(tau_real, label=r'$\tau~_{ real}$', histtype=u'step',
         density=True, linewidth=2.5, bins=bins)
plt.legend(loc='upper right')
plt.title('CATE(Conditional average treatment effect)')
plt.xlabel(r'$\tau$', fontsize=14)
plt.ylabel('Density')

plt.subplot(1, 2, 2)
bins = np.linspace(min(min(mu0_pred),min(mu0_real)),
                    max(max(mu0_pred),max(mu0_real)), 15)
plt.hist(mu0_pred, alpha=0.7, label=r'$\mu_{0~pred}$',
         density=True, bins=bins)
plt.hist(mu0_real,label=r'$\mu_{0~real}$', histtype=u'step',
         density=True, linewidth=2.5, bins=bins)
plt.legend(loc='upper right')
plt.title(r'$\mu_0(x)$')
plt.xlabel(r'$\mu_0$', fontsize=14)
plt.ylabel('Density')

plt.tight_layout()
plt.show()
```
To run the same example directly in R, additional R packages need to be installed: reticulate, repr and gensvm. Then run the code below: 
```R
library(reticulate)
library(gensvm)
library(repr)
causalNets <- import("causal_nets")
np <-import("numpy")
set.seed(3)

N <- 10000
d <- 10
X <- matrix(runif(N * d), N, d)
mu0_real <- 0.012*X[, 4] - 0.75*X[, 6]*X[ ,8] - 0.9*X[, 5] - rowMeans(X)
tau_real <- X[, 3] + 0.04*X[, 10] - 0.35*log(X[, 4]) 
prob_of_T <- 0.5
T <- rbinom(N, 1, prob_of_T)
normal_errors <- rnorm(N)
Y <- mu0_real + tau_real*T + normal_errors

split = gensvm.train.test.split(X, train.size=0.8, shuffle=TRUE,
                                random.state=42, return.idx=TRUE)
# Splitting the data
X_train = X[split$idx.train,]
Y_train = Y[split$idx.train]
T_train = T[split$idx.train]
X_valid = X[split$idx.test,]
Y_valid = Y[split$idx.test]
T_valid = T[split$idx.test]

# Converting arrays into ndarrays which are recognized by Python
X_train = np$array(X_train)
T_train = np$array(T_train)
Y_train = np$array(Y_train)
X_valid = np$array(X_valid)
T_valid = np$array(T_valid)
Y_valid = np$array(Y_valid)
X = np$array(X)
T = np$array(T)
Y = np$array(Y)

# Getting causal estimates 
coeffs = causalNets$causal_net_estimate(
    list(X_train, T_train, Y_train), list(X_valid, T_valid, Y_valid),
    list(X, T, Y),  list(30L, 20L, 15L, 10L, 5L), dropout_rates=NULL, batch_size=NULL,
    alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009, max_epochs_without_change=30L,
    max_nepochs=10000L, seed=NULL, estimate_ps=FALSE, verbose=TRUE)

# Plotting estimated coefficient vs true coefficients
tau_pred <- as.vector(coeffs[[1]])
mu0_pred <- as.vector(coeffs[[2]])

c1 <- rgb(0, 0, 230, max=255, alpha=70, names="purple")
c2 <- rgb(0, 230, 0, max=255, alpha=70, names="green") 

options(repr.plot.width=9, repr.plot.height=5)
par(mfrow=c(1,2))

# Plot histograms for tau_pred and tau_real
breaks <- seq(min(c(tau_pred, tau_real)), max(c(tau_pred, tau_real)), length.out=15)

hist(tau_pred, main="CATE(Conditional average treatment effect)",
     freq=FALSE, cex.main=0.9, ylab="Density", xlab=expression(tau), col=c1,
     border=F, breaks=breaks)
hist(tau_real, col=c2, add=TRUE, freq=FALSE, border=F, breaks=breaks)
legend("topright", legend=c(expression(tau[' pred']), expression(tau[' real'])), 
       fill=c(c1, c2), box.lty=0, cex=0.8, border=F, x.intersp=0.5)

# Plot histograms for mu0_pred and mu0_real
breaks <- seq(min(c(mu0_pred, mu0_real)), max(c(mu0_pred, mu0_real)), length.out=15)

hist(mu0_pred, main=expression(mu[0]*"(x)"),
     freq=FALSE, cex.main=0.9, ylab="Density", xlab=expression(mu[0]), col=c1,
     border=F, breaks=breaks)
hist(mu0_real, col=c2, add=TRUE, freq=FALSE, border=F, breaks=breaks)
legend("topright", legend=c(expression(mu[0][' pred']), expression(mu[0][' real'])), 
       fill=c(c1, c2), box.lty=0, cex=0.8, border=F, x.intersp=0.5)
```

### Explanation of the parameters of the main function causal_net_estimate()
```
causal_net_estimate(training_data, validation_data, test_data, hidden_layer_sizes,
                    dropout_rates=None, batch_size=None, alpha=0., r_par=0.,
                    optimizer='Adam', learning_rate=0.0009, max_epochs_without_change=30,
                    max_nepochs=5000, seed=None, estimate_ps=False, verbose=True,
                    hidden_layer_sizes_t=None, dropout_rates_t=None, batch_size_t=None,
                    alpha_t=0., r_par_t=0., optimizer_t='Adam', learning_rate_t=0.0009,
                    max_epochs_without_change_t=30, max_nepochs_t=5000, seed_t=None)
```

    Parameters
    ----------
    training_data: list of arrays
        Data on which the training of the Neural Network will be
        performed. It must be comprised as a list of arrays, in the
        following manner: [X_train, T_train, Y_train]
        Here, `X_train` is an array of input features, `T_train` is
        the treatment array, and `Y_train` is the target array.
    validation_data: list of arrays
        Data on which the validation of the Neural Network will be
        performed. It has to be composed in the same manner as the
        training data.
    test_data: list of arrays
        Data on which we want to perform estimation. It has to be
        composed in the same manner as the training and validation data.
    hidden_layer_sizes: list of ints
        `hidden_layer_sizes` is a list that defines a size and width of
        the neural network that estimates causal coefficients. Length of
        the list defines the number of hidden layers. Entries of the
        list define the number of hidden units in each hidden layer.
        No default value, it needs to be provided.
        E.g. hidden_layer_sizes = [60, 30]
    dropout_rates: list of floats or None, optional
        If it's a list than the values of the list represent dropout
        rate for each layer of the feed-forward neural network
        that estimates causal coefficients. Each entry of the list has
        to be between 0 and 1. Also, list has to be of the same length
        as the list 'hidden_layer_sizes'. If is set to None, than
        dropout is not applied. Default value is None.
    batch_size: int, optional
        Batch size for the neural network that estimates causal
        coefficients. Default value is None. If batch_size is None,
        than batch size is equal to length of the training dataset for
        training datasets smaller than 50000 rows and set to 1024 for
        larger datasets. Otherwise, it is equal to the value provided.
    alpha: float, optional
        Regularization strength parameter for the neural network that
        estimates causal coefficients. Default value is 0.
    r_par: float, optional
        Mixing ratio of Ridge and Lasso regression for the neural
        network that estimates causal coefficients.
        Has to be between 0 and 1. If r_par = 1, than this is equal to
        having Lasso regression. If r_par = 0, than it is equal to
        having Ridge regression. Default value is 0.
    optimizer: {'Adam', 'GradientDescent', 'RMSprop'}, optional
        Which optimizer to use for the neural network that estimates
        causal coefficients. Default: 'Adam'.
    learning_rate: scalar, optional
        Learning rate for the neural network that estimates
        causal coefficients. Default value is 0.0009.
    max_epochs_without_change: int, optional
        Number of epochs with no improvement on the validation loss to
        wait before stopping the training for the neural network that
        estimates causal coefficients. Default value is 30.
    max_nepochs: int, optional
        Maximum number of epochs for which neural network that
        estimates causal coefficients will be trained.
        Default value is 5000.
    seed: int or None, optional
        Tensorflow seed number for the neural network that estimates
        causal coefficients. Default value is None.
    estimate_ps: bool, optional
        Should the propensity scores be estimated or not. If the
        treatment is randomized then this variable should be set to
        False. In not randomized treatment case, it should be set to
        True. Default value is False.
    verbose: bool, optional
        Should the model summary and losses during training be printed.
        If it is set to False, the printing behavior is suppressed.
        Default value is True.
    hidden_layer_sizes_t: list of ints or None, optional
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
    alpha_t: float, optional
        Regularization strength parameter for the neural network that
        estimates propensity scores. Default value is 0.
    r_par_t: float, optional
        Mixing ratio of Ridge and Lasso regression for the neural
        network that estimates propensity scores.
        Has to be between 0 and 1. If r_par_t = 1, than this is equal to
        having Lasso regression. If r_par_t = 0, than it is equal to
        having Ridge regression. Default value is 0.
    optimizer_t: {'Adam', 'GradientDescent', 'RMSprop'}, optional
        Which optimizer to use for the neural network that estimates
        propensity scores. Default: 'Adam'.
    learning_rate_t: scalar, optional
        Learning rate for the neural network that estimates propensity
        scores. Default value is 0.0009.
    max_epochs_without_change_t: int, optional
        Number of epochs with no improvement on the validation loss to
        wait before stopping the training for the neural network that
        estimates propensity scores. Default value is 30.
    max_nepochs_t: int, optional
        Maximum number of epochs for which neural network, that
        estimates propensity scores, will be trained.
        Default value is 5000.
    seed_t: int or None, optional
        Tensorflow seed number for the neural network that estimates
        propensity scores. Default value is None.

    Returns
    -------
    tau_pred: ndarray
        Estimated conditional average treatment effect.
    mu0_pred: ndarray
        Estimated target value given x in case of no treatment.
    prob_t_pred: ndarray
        Estimated propensity scores.
    psi_0: ndarray
        Influence function for given x in case of no treatment.
    psi_1: ndarray
        Influence function for given x in case of treatment.
    history_dict: dict
        Dictionary that stores validation and training loss values for
        CoeffNet.
    history_ps_dict: dict
        Dictionary that stores validation and training loss values for
        PropensityScoreNet. If estimate_ps is set to None,
        history_ps_dict is set to None as well.

### References

Farrell, M.H., Liang, T. and Misra, S. (2021), Deep Neural Networks for Estimation and Inference. Econometrica, 89: 181-213. https://doi.org/10.3982/ECTA16901
