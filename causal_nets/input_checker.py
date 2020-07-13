import numpy as np


class InputChecker():
    '''
    Checks the validity of input variables inserted in the
    causal_net_estimate function.

    Parameters
    ----------
    Same parameters as for the main function causal_net_estimate, located
    in utils.py.

    Raises
    ------
    Error: In case of mismatch between the provided input data and what
    the causal_net_estimate function expects.
    '''
    def __init__(self, training_data, validation_data, test_data,
                 hidden_layer_sizes, dropout_rates, batch_size, alpha, r_par,
                 optimizer, learning_rate, max_epochs_without_change,
                 max_nepochs, seed, estimate_ps, hidden_layer_sizes_t,
                 dropout_rates_t, batch_size_t, alpha_t, r_par_t, optimizer_t,
                 learning_rate_t, max_epochs_without_change_t, max_nepochs_t,
                 seed_t):

        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = dropout_rates
        self.batch_size = batch_size
        self.alpha = alpha
        self.r_par = r_par
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_epochs_without_change = max_epochs_without_change
        self.max_nepochs = max_nepochs
        self.seed = seed
        self.estimate_ps = estimate_ps
        self.hidden_layer_sizes_t = hidden_layer_sizes_t
        self.dropout_rates_t = dropout_rates_t
        self.batch_size_t = batch_size_t
        self.alpha_t = alpha_t
        self.r_par_t = r_par_t
        self.optimizer_t = optimizer_t
        self.learning_rate_t = learning_rate_t
        self.max_epochs_without_change_t = max_epochs_without_change_t
        self.max_nepochs_t = max_nepochs_t
        self.seed_t = seed_t

    def _check_if_the_array_is_a_vector(self, data_array):
        '''Check if the data_array is an array of shape Nx1 or N.'''
        data_array = np.array(data_array)
        return len(data_array) == len(data_array.reshape(-1))

    def _is_data_list_valid(self, data_list):
        '''Check if the inputs for training, validation and test data
        are valid.'''
        if not isinstance(data_list, list):
            raise TypeError('Data such as training, validation and' +
                            ' test must be provided as lists')

        if not len(data_list) == 3:
            raise Exception('All data lists must contain array of input' +
                            ' features, treatment array and target array')

        if not len(data_list[0]) == len(data_list[1]) == len(data_list[2]):
            raise Exception('Length of input features array, treatment array' +
                            ' and target array must be equal')

        if not(self._check_if_the_array_is_a_vector(data_list[1]) and
               self._check_if_the_array_is_a_vector(data_list[2])):
            raise Exception(
                'Treatment and target arrays must be provided as arrays' +
                ' of shape Nx1 or N')

    def _is_treatment_array_binary(self, treatment_array):
        '''Check if the treatment array is binary.'''
        if not list(np.unique(treatment_array)) == [0, 1]:
            raise Exception('Treatment array needs to be a binary' +
                            ' array of 0s and 1s')

    def _are_hidden_layer_sizes_valid(self, hidden_layer_sizes):
        '''Check if the input for hidden_layer_sizes is valid.'''
        if not isinstance(hidden_layer_sizes, list):
            raise TypeError('hidden_layer_sizes must be provided as a list')
        else:
            for i in hidden_layer_sizes:
                if not isinstance(i, int):
                    raise TypeError(
                        'Each hidden layer in hidden_layer_sizes list must' +
                        ' be provided as an integer')
                else:
                    if i <= 0:
                        raise ValueError(
                            'Each hidden layer in hidden_layer_sizes list' +
                            ' has to be a positive integer')

    def _are_dropout_rates_valid(self, dropout_rates, hidden_layer_sizes):
        '''Check if the input for dropout_rates is valid.'''
        if dropout_rates and not isinstance(dropout_rates, list):
            raise TypeError(
                'dropout_rates must be provided as a list or set to None')

        if dropout_rates is not None:
            for i in dropout_rates:
                if not isinstance(i, (float, int)):
                    raise TypeError(
                        'Dropout rates must be provided as numeric values')
                else:
                    if i < 0 or i > 1:
                        raise ValueError(
                            'Dropout rates for each hidden layer must' +
                            ' be a number between 0 and 1')
            if not len(dropout_rates) == len(hidden_layer_sizes):
                raise Exception('dropout_rates list has to be of the same' +
                                ' length as the hidden_layer_sizes list')

    def _is_batch_size_valid(self, batch_size):
        '''Check if the input for batch_size is valid.'''
        if batch_size and not isinstance(batch_size, int):
            raise TypeError('Batch size must be an integer or set to None')

        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError('Batch size must be a positive integer')

            if batch_size > len(self.training_data[0]):
                raise Exception('Batch size value should be less then the' +
                                ' length of the training data')

    def _is_alpha_parameter_valid(self, alpha):
        '''Check if the input for regularization strength parameter,
           alpha, is valid'''
        if not isinstance(alpha, float):
            raise TypeError('Alpha parameter must be provided as a float')
        else:
            if alpha < 0:
                raise ValueError(
                    'Alpha parameter must be a non-negative float')

    def _is_r_par_valid(self, r_par):
        '''Check if the input for r_par is valid.'''
        if not isinstance(r_par, float):
            raise TypeError('r_par must be provided as a float')
        else:
            if r_par < 0 or r_par > 1:
                raise ValueError('r_par must be a float between 0 and 1')

    def _is_optimizer_valid(self, optimizer):
        '''Check if the input for optimizer is valid.'''
        if not isinstance(optimizer, str):
            raise TypeError('Optimizer must be provided as a string')
        else:
            if optimizer not in ['Adam', 'GradientDescent', 'RMSprop']:
                raise ValueError(
                    "Optimizer not recognized. It must be one of the three" +
                    " options: 'Adam', 'GradientDescent', or 'RMSprop'")

    def _is_learning_rate_valid(self, learning_rate):
        '''Check if the input for learning rate is valid.'''
        if not isinstance(learning_rate, float):
            raise TypeError('Learning rate must be provided as a float')
        else:
            if learning_rate <= 0:
                raise ValueError('Learning rate must be a positive float')

    def _is_max_epochs_valid(self, max_epochs):
        '''Check if the input for max_nepochs is valid.'''
        if not isinstance(max_epochs, int):
            raise TypeError('max_nepochs must be provided as an integer')
        else:
            if max_epochs <= 0:
                raise ValueError('max_nepochs must be a positive integer')

    def _is_max_epochs_without_change_valid(self, max_epochs_without_change):
        '''Check if the input for max_epochs_without_change is valid.'''
        if not isinstance(max_epochs_without_change, int):
            raise TypeError(
                'max_epochs_without_change must be provided as an integer')
        else:
            if max_epochs_without_change < 0:
                raise ValueError(
                    'max_epochs_without_change must be a non-negative integer')

    def _is_seed_par_valid(self, seed):
        '''Check if the value provided for seed parameter is valid.'''
        if seed and not isinstance(seed, int):
            raise TypeError(
                'Seed must be provided as an integer or set to None')
        if seed is not None:
            if seed <= 0:
                raise ValueError('Seed must be a positive integer')

    def _is_estimate_ps_valid(self, estimate_ps):
        '''Check if the value provided for estimate_ps is valid.'''
        if not isinstance(estimate_ps, bool):
            raise TypeError('estimate_ps must be provided as a boolean')

    def check_all_parameters(self):
        '''Checks the validity of all parameters provided to
        causal_net_estimate function'''

        # Check the validity of the data provided:
        self._is_data_list_valid(self.training_data)
        self._is_data_list_valid(self.validation_data)
        self._is_data_list_valid(self.test_data)

        # Check that the treatment is binary
        self._is_treatment_array_binary(self.training_data[1])
        self._is_treatment_array_binary(self.validation_data[1])
        self._is_treatment_array_binary(self.test_data[1])

        # Check the validity of inputs for the first neural network:
        self._are_hidden_layer_sizes_valid(self.hidden_layer_sizes)
        self._are_dropout_rates_valid(self.dropout_rates,
                                      self.hidden_layer_sizes)
        self._is_batch_size_valid(self.batch_size)
        self._is_alpha_parameter_valid(self.alpha)
        self._is_r_par_valid(self.r_par)
        self._is_optimizer_valid(self.optimizer)
        self._is_learning_rate_valid(self.learning_rate)
        self._is_max_epochs_valid(self.max_nepochs)
        self._is_max_epochs_without_change_valid(self.max_epochs_without_change)
        self._is_seed_par_valid(self.seed)

        # Check the validity of estimate_ps:
        self._is_estimate_ps_valid(self.estimate_ps)

        # Check the validity of inputs for the second neural network:
        if self.estimate_ps:
            self._are_hidden_layer_sizes_valid(self.hidden_layer_sizes_t)
            self._are_dropout_rates_valid(self.dropout_rates_t,
                                          self.hidden_layer_sizes_t)
            self._is_batch_size_valid(self.batch_size_t)
            self._is_alpha_parameter_valid(self.alpha_t)
            self._is_r_par_valid(self.r_par_t)
            self._is_optimizer_valid(self.optimizer_t)
            self._is_learning_rate_valid(self.learning_rate_t)
            self._is_max_epochs_valid(self.max_nepochs_t)
            self._is_max_epochs_without_change_valid(
                self.max_epochs_without_change_t)
            self._is_seed_par_valid(self.seed_t)
