
import os
# Stopping Tensorflow from printing info messages
# and warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import Callback
import logging
import random
import warnings
import matplotlib.cbook

# Stopping deprecation warnings
logging.getLogger('tensorflow').disabled = True
warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)


def _plotting_loss_functions(history):
    '''
    Plotting loss functions.

    Parameters
    ----------
    history:
        Keras history callback that records the run events and stores
        them in history object.
    '''
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    epochs = range(1, len(loss_val) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss_train, 'ro', alpha=0.7, label='Training loss')
    _ = plt.plot(epochs, loss_val, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss value')
    plt.legend()


class _MyLogger(Callback):
    '''
    Printing validation loss after the first epoch and
    then after every n epochs.

    Parameters
    ----------
     n_epochs: after how many epochs to print a validation loss
    '''

    def __init__(self, n_epochs):
        super().__init__()
        self.after_n_epochs = n_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.after_n_epochs == 0:
            print('%d val loss: %.4f\n' % (epoch+1, logs['val_loss']))


class CoeffNet():
    '''
    Neural network for causal effect estimation.

    Parameters
    ----------
    hidden_layer_sizes: list of ints
        Length of the list defines the number of hidden layers.
        Entries of the list define the number of hidden units in each
        hidden layer. (e.g. hidden_layer_sizes = [60, 30])
    dropout_rates: list of floats
        Dropout rate for each layer. Each entry has to be between
        0 and 1. Has to be of length len(hidden_layer_sizes).
    batch_size: int
        Batch size.
    alpha_l1: float
        Lasso(L1) regularization factor.
    alpha_l2: float
        Ridge(L2) regularization factor.
    optimizer: str
        Which optimizer to use.
        See page: https://keras.io/optimizers/ for optimizers options.
    learning_rate: scalar
        Learning rate.
    max_epochs_without_change: int
        Number of epochs with no improvement on the validation loss to
        wait before stopping the training.
    max_nepochs: int
        Maximum number of epochs for which neural network will
        be trained.
    distribution: {'Sigmoid', 'LinearRegression'}
        If the target variable is a binary categorical, then use
        distribution='Sigmoid'. Otherwise, if the target variable is
        continuous then distribution='LinearRegression'.
    nparameters: int
        Number of units in the output layer.
    '''
    def __init__(self, hidden_layer_sizes, dropout_rates,
                 batch_size, alpha_l1, alpha_l2, optimizer, learning_rate,
                 max_epochs_without_change, max_nepochs, distribution):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = dropout_rates
        self.batch_size = batch_size
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs_without_change = max_epochs_without_change
        self.max_nepochs = max_nepochs
        self.distribution = distribution
        self.nparameters = 2

    def _last_layer(self, combined_input):
        '''
        Building a custom layer which will be appended at the end of
        feed-forward neural network.

        In the case of regression, this layer will return the value of:
        tau * T + mu0
        and for sigmoid:
        sigmoid(tau * T + mu0)
        where `tau` is conditional average treatment effect for each
        individual, `T` is the treatment for each individual and
        `mu0` is estimated target value given x in case of no treatment
        for each individual.

        Parameters
        ----------
        combined_input: Tensor
            Concatenated layer of `tau`, `mu0` and `input_t`.
        '''
        import tensorflow as tf
        tau = combined_input[:, 0:1]
        mu0 = combined_input[:, 1:2]

        t = combined_input[:, self.nparameters:]

        V_values = tf.multiply(t, tau) + mu0

        if self.distribution == 'Sigmoid':
            return tf.sigmoid(V_values)
        elif self.distribution == 'LinearRegression':
            return V_values

    def _last_layer_output_shape(self, input_shape):
        '''
        Returns the shape of the custom last layer.

        Parameters
        ----------
        input_shape: 'Shape of the previous layer'

        Returns:
        -------
        The shape of the custom last layer
        '''
        shape = list(input_shape)
        assert len(shape) == 2
        shape[-1] = 1
        return tuple(shape)

    def _building_the_model(self, nfeatures):
        '''
        Build the whole fully connected neural network that estimates
        causal coefficients.

        Parameters
        ----------
        nfeatures: int
            Number of features in the input layer.

        Returns
        -------
        model: keras model
            Full keras model
        betas_model: keras model
            Model, that encapsulates only the feed-forward neural
            network, which outputs coefficients.
        '''
        # Matrix of consumer characteristics
        input_x = Input(shape=(nfeatures,))

        # Array of treatments
        input_t = Input(shape=(1,))

        for i in range(len(self.hidden_layer_sizes)):
            if i == 0:
                output = Dropout(self.dropout_rates[i])(input_x)
            else:
                output = Dropout(self.dropout_rates[i])(output)

            if (self.alpha_l1 is None and self.alpha_l2 is None):
                reg = None
            else:
                reg = keras.regularizers.l1_l2(l1=self.alpha_l1,
                                               l2=self.alpha_l2)

            output = Dense(self.hidden_layer_sizes[i], activation='relu',
                           use_bias=True, kernel_initializer='glorot_uniform',
                           kernel_regularizer=reg,
                           bias_initializer='zeros')(output)

        betas = Dense(self.nparameters, activation=None, use_bias=True,
                      kernel_initializer='glorot_uniform',
                      kernel_regularizer=reg, bias_initializer='zeros')(output)

        combined = concatenate([betas, input_t], axis=-1)

        output_tensor = Lambda(
            self._last_layer,
            output_shape=self._last_layer_output_shape)(combined)

        model = Model(inputs=[input_x, input_t], outputs=output_tensor)
        betas_model = Model(inputs=input_x, outputs=betas)

        if self.optimizer == 'Adam':
            opt = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9,
                                        beta_2=0.999, epsilon=None, decay=0.0,
                                        amsgrad=True)

        elif self.optimizer == 'GradientDescent':
            opt = keras.optimizers.SGD(lr=self.learning_rate, momentum=0.0,
                                       decay=0.0, nesterov=False)
        else:
            opt = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9,
                                           epsilon=None, decay=0.0)

        if self.distribution == 'Sigmoid':
            model.compile(optimizer=opt,
                          loss='binary_crossentropy')
        elif self.distribution == 'LinearRegression':
            model.compile(optimizer=opt,
                          loss='mean_squared_error')
        else:
            raise ValueError('Distribution not recognized!')
        return model, betas_model

    def training_NN(self, training_data, validation_data):
        '''
        Train a NN for max_nepochs or until early stopping criterion
        is met.

        Parameters
        ----------
        training_data: list of arrays
            Data on which the training of the Neural Network will be
            performed. It is comprised as a list of arrays, in the
            following manner:
            [X_train, T_train, Y_train], where `X_train` is an array of
            input features, `T_train` is the treatment array, and
            `Y_train` is the target array.

        validation_data: list of arrays
            Data on which the validation of the Neural Network will be
            performed. It is composed in the same manner as the
            training data.

        Returns
        -------
        betas_model: keras model
            Model, that encapsulates only the feed-forward neural
            network, which as an output has causal coefficients.
        '''
        # Clearing the weights
        K.clear_session()

        nfeatures = np.shape(training_data[0])[1]
        # Building the modeL
        model, betas_model = self._building_the_model(nfeatures)

        model.summary()

        EarlyStop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.max_epochs_without_change,
            restore_best_weights=True)

        # Training the model
        history = model.fit(
            x=training_data[0:2], y=training_data[2],
            epochs=self.max_nepochs, batch_size=self.batch_size,
            validation_data=(validation_data[0:2], validation_data[2]),
            callbacks=[EarlyStop, _MyLogger(50)], shuffle=True, verbose=0)

        # Plotting loss functions
        _plotting_loss_functions(history)
        return betas_model

    def retrieve_coeffs(self, betas_model, input_value, plot_coeffs):
        '''
        After training is completed retrieve coefficient values.

        Parameters
        ----------
        betas_model: keras model
            Model, that encapsulates only the feed-forward neural
            network, which as an output has causal coefficients.
        input_value: array like
            Features array
        plot_coeffs: bool
            Should predicted coefficients be plotted or not.

        Returns
        -------
        tau_pred: ndarray
            Estimated conditional average treatment effect.
        mu0_pred: ndarray
            Estimated target value given x in case of no treatment.
        '''
        betas_pred = betas_model.predict(input_value)
        tau_pred = betas_pred[:, :1]
        mu0_pred = betas_pred[:, 1:2]
        if plot_coeffs:
            self._plot_all_coeffs(tau_pred, mu0_pred)
        return tau_pred, mu0_pred

    def _plot_coeff_distribution(self, coeff_pred, name):
        '''
        Plot the distribution of the given coefficient.

        Parameters
        ----------
        coeff_pred: ndarray
            Coefficient array which distribution
            we want to plot.
        '''
        plt.hist(coeff_pred, alpha=0.7, label=name+'_pred')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.title(name)

    def _plot_all_coeffs(self, tau_pred, mu0_pred):
        '''
        Plotting predicted treatment effect coefficients'
        distributions.

        Parameters
        ----------
        tau_pred: ndarray
            Estimated conditional average treatment effect.
        mu0_pred: ndarray
            Estimated target value given x in case of no treatment.
        '''
        plt.figure(figsize=(8, 4))
        plt.clf()

        plt.subplot(1, 2, 1)
        self._plot_coeff_distribution(tau_pred, 'tau')

        plt.subplot(1, 2, 2)
        self._plot_coeff_distribution(mu0_pred, r'$\mu_0$')

        plt.tight_layout()


class PropensityScoreNet():
    '''
    Neural network for propensity scores estimation.

    Parameters
    ----------
    Same as in class CoeffNet.
    '''
    def __init__(self, hidden_layer_sizes, dropout_rates, batch_size,
                 alpha_l1, alpha_l2, optimizer, learning_rate,
                 max_epochs_without_change, max_nepochs):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = dropout_rates
        self.batch_size = batch_size
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs_without_change = max_epochs_without_change
        self.max_nepochs = max_nepochs
        self.nparameters = 1

    def _building_the_model(self, nfeatures):
        '''
        Build the whole fully connected neural network that estimates
        propensity scores.

        Parameters
        ----------
        nfeatures: int
            Number of features in the input layer.

        Returns
        -------
        model: keras model
            Keras model which returns propensity scores.
        '''
        # Matrix of consumer characteristics
        input_x = Input(shape=(nfeatures,))

        for i in range(len(self.hidden_layer_sizes)):
            if i == 0:
                output = Dropout(self.dropout_rates[i])(input_x)
            else:
                output = Dropout(self.dropout_rates[i])(output)

            if (self.alpha_l1 is None and self.alpha_l2 is None):
                reg = None
            else:
                reg = keras.regularizers.l1_l2(l1=self.alpha_l1,
                                               l2=self.alpha_l2)

            output = Dense(self.hidden_layer_sizes[i], activation='relu',
                           use_bias=True, kernel_initializer='glorot_uniform',
                           kernel_regularizer=reg,
                           bias_initializer='zeros')(output)

        ps_outputs = Dense(self.nparameters, activation='sigmoid',
                           use_bias=True, kernel_initializer='glorot_uniform',
                           kernel_regularizer=reg,
                           bias_initializer='zeros')(output)

        model = Model(inputs=input_x, outputs=ps_outputs)

        if self.optimizer == 'Adam':
            opt = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9,
                                        beta_2=0.999, epsilon=None, decay=0.0,
                                        amsgrad=True)

        elif self.optimizer == 'GradientDescent':
            opt = keras.optimizers.SGD(lr=self.learning_rate, momentum=0.0,
                                       decay=0.0, nesterov=False)
        else:
            opt = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9,
                                           epsilon=None, decay=0.0)

        model.compile(optimizer=opt, loss='binary_crossentropy')
        return model

    def training_NN(self, training_data, validation_data):
        '''
        Train a NN for max_nepochs or until early stopping criterion
        is met.

        Parameters
        ----------
        training_data: list of arrays
            Data on which the training of the Neural Network will be
            performed. It is comprised as a list of arrays, in the
            following manner:
            [X_train, T_train], where `X_train` is an array of
            input features, `T_train` is the treatment array.

        validation_data: list of arrays
            Data on which the validation of the Neural Network will be
            performed. It is composed in the same manner as the
            training data.

        Returns
        -------
        model: keras model
            Keras model which returns propensity scores.
        '''
        # Clearing the weights
        K.clear_session()

        nfeatures = np.shape(training_data[0])[1]
        # Building the modeL
        model = self._building_the_model(nfeatures)

        model.summary()

        EarlyStop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.max_epochs_without_change,
            restore_best_weights=True)

        # Training the model
        history = model.fit(
            x=training_data[0], y=training_data[1],
            epochs=self.max_nepochs, batch_size=self.batch_size,
            validation_data=(validation_data[0], validation_data[1]),
            callbacks=[EarlyStop, _MyLogger(50)], shuffle=True, verbose=0)

        # Plotting loss functions
        _plotting_loss_functions(history)
        return model

    def retrieve_propensity_scores(self, model, input_value,
                                   plot_coeffs):
        '''
        After training is completed retrieve propensity scores for
        a given data.

        Parameters
        ----------
        model: keras model
            Keras model which returns propensity scores.
        input_value: array like
            Features array
        plot_coeffs: bool
            Should histogram of predicted propensity scores
            be plotted or not.

        Returns
        -------
        prob_of_t_pred: ndarray
            Estimated propensity scores
        '''
        prob_of_t_pred = model.predict(input_value)
        if plot_coeffs:
            self._plot_propensity_scores(prob_of_t_pred)
        return prob_of_t_pred

    def _plot_propensity_scores(self, prob_of_t_pred):
        '''
        Plot a histogram for estimated propensity scores.

        Parameters
        ----------
        prob_of_t_pred: ndarray
            Estimated conditional average treatment effect.
        mu0_pred: ndarray
            Estimated target value given x in case of no treatment.
        Plotting predicted treatment effect coefficients.
        '''
        plt.figure(figsize=(8, 4))
        plt.clf()

        plt.hist(prob_of_t_pred, alpha=0.7)
        plt.ylabel('Density')
        plt.xlabel('Propensity scores')
        plt.legend(loc='upper right')
        plt.title('Estimated propensity scores')


def determine_batch_size(batch_size, training_data):
    '''
    Assign batch size value if the batch size is not provided, or if
    it is check if it is valid one and return the same value if it is.

    If batch_size is None, than batch size is equal to length of
    training dataset for training datasets smaller than 50000 rows.
    Otherwise, it is set to 1024.

    Parameters
    ----------
    batch_size: int or None
        Batch size.
    training_data:
        Data on which training of the neural network should be
        performed. Provided as a list of arrays.

    Returns
    -------
    batch_size: int
        Batch size.
    '''
    if batch_size is None:
        if len(training_data[0]) > 50000:
            batch_size = 1024
        else:
            batch_size = len(training_data[0])
    else:
        assertion_error = ('Batch size value should be less then the' +
                           ' length of the training data!')
        assert batch_size < len(training_data[0]), assertion_error
                
    return batch_size


def determine_dropout_rates(hidden_layer_sizes):
    '''
    Sets all dropout rates to zero if dropout rates are not provided.

    Parameters
    ----------
    hidden_layer_sizes: list of ints
        Length of the list defines the number of hidden layers.
        Entries of the list define the number of hidden units in each
        hidden layer.

    Returns
    -------
    Dropout rates list of appropriate length with all values
    set to zero.
    '''
    return [0]*len(hidden_layer_sizes)


def _doubly_robust_estimator(mu0_pred, tau_pred, Y, T,
                             prob_t_pred, estimate_ps):
    '''
    Calculate the target value for each individual when treatment is
    0 or 1.

    Parameters
    ----------
        mu0_pred: ndarray
            Estimated target value given x in case of no treatment.
        tau_pred: ndarray
            Estimated conditional average treatment effect.
        Y: ndarray
            Target value array.
        T: ndarray
            Treatment array.
        prob_t_pred: ndarray
            Estimated propensity scores.
        estimate_ps: bool
            Should the propensity scores be estimated or not.
    Returns
    -------
        psi_0: ndarray
            Doubly robust estimate of target value given x in case of
            no treatment.
        psi_1: ndarray
            Doubly robust estimate of target value given x in case of
            treatment.
    '''
    T = np.array(T).reshape(-1,1)
    Y = np.array(Y).reshape(-1,1)

    first_part = (1-T) * (Y-mu0_pred)
    second_part = T * (Y-mu0_pred-tau_pred)

    if estimate_ps:
        prob_t_pred[prob_t_pred < 0.001] = 0.001
        prob_t_pred[prob_t_pred > 0.999] = 0.999
        psi_0 = (first_part/(1-prob_t_pred)) + mu0_pred
        psi_1 = (second_part/prob_t_pred) + mu0_pred + tau_pred
    else:
        psi_0 = (first_part/(1-np.mean(T))) + mu0_pred
        psi_1 = (second_part/np.mean(T)) + mu0_pred + tau_pred
    return psi_0, psi_1


def causal_net_estimate(ind_X, ind_T, ind_Y, training_data, validation_data,
                        estimation_data,  hidden_layer_sizes, dropout_rates=None,
                        batch_size=None, alpha_l1=0., alpha_l2=0.,
                        optimizer='Adam', learning_rate=0.0009,
                        max_epochs_without_change=30, max_nepochs=5000,
                        distribution='LinearRegression', estimate_ps=False,
                        hidden_layer_sizes_t=None, dropout_rates_t=None,
                        batch_size_t=None, alpha_l1_t=0., alpha_l2_t=0.,
                        optimizer_t='Adam', learning_rate_t=0.0009,
                        max_epochs_without_change_t=30, max_nepochs_t=5000,
                        plot_coeffs=True):
    '''
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
    estimation_data: list of arrays
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
    estimate_ps: False, optional
        Should the propensity scores be estimated or not. If the
        treatment is randomized then this variable should be set to 
        False. In not randomized treatment case, it should be set to 
        True. Default value is False.
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
        Estimated target value given x in case of no treatment.
    prob_t_pred: ndarray
        Estimated propensity scores
    psi_0: ndarray
        Doubly robust estimate of target value given x in case of
        no treatment.
    psi_1: ndarray
        Doubly robust estimate of target value given x in case of
        treatment.
    '''
    batch_size = determine_batch_size(batch_size, training_data)
    if dropout_rates is None:
        dropout_rates = determine_dropout_rates(hidden_layer_sizes)

    coeff_net = CoeffNet(hidden_layer_sizes, dropout_rates, batch_size,
                         alpha_l1, alpha_l2, optimizer, learning_rate,
                         max_epochs_without_change, max_nepochs, distribution)

    model_coeff_net = coeff_net.training_NN(
        [training_data[ind_X], training_data[ind_T], training_data[ind_Y]],
        [validation_data[ind_X], validation_data[ind_T],
         validation_data[ind_Y]])

    tau_pred, mu0_pred = coeff_net.retrieve_coeffs(
        model_coeff_net, estimation_data[ind_X], plot_coeffs=plot_coeffs)

    if estimate_ps:
        if batch_size_t is None:
            batch_size_t = determine_batch_size(batch_size_t, training_data)
        if dropout_rates_t is None:
            dropout_rates_t = determine_dropout_rates(hidden_layer_sizes_t)
        if hidden_layer_sizes_t is None:
            raise ValueError('Hidden layer sizes needs to be specified for' +
                             'the second neural network as well')

        ps_net = PropensityScoreNet(
            hidden_layer_sizes_t, dropout_rates_t, batch_size_t, alpha_l1_t,
            alpha_l2_t, optimizer_t, learning_rate_t,
            max_epochs_without_change_t, max_nepochs_t)

        model_ps_net = ps_net.training_NN(
            [training_data[ind_X], training_data[ind_T]],
            [validation_data[ind_X], validation_data[ind_T]])

        prob_t_pred = ps_net.retrieve_propensity_scores(
            model_ps_net, estimation_data[ind_X], plot_coeffs=plot_coeffs)
    else:
        prob_t_pred = np.mean(estimation_data[ind_T])
    
    psi_0, psi_1 = _doubly_robust_estimator(
        mu0_pred, tau_pred, estimation_data[ind_Y], estimation_data[ind_T],
        prob_t_pred, estimate_ps)

    return tau_pred, mu0_pred, prob_t_pred, psi_0, psi_1
