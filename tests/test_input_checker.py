import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from causal_nets.input_checker import InputChecker
import sys


@pytest.fixture(scope='class')
def inp_check(request):
    '''Initialize one instance of InputChecker class that will be
    used later to perform different tests.'''
    # Setting the seeds
    np.random.seed(3)

    # Generating the fake data
    N = 100
    X = np.random.uniform(low=0, high=1, size=[N, 10])
    mu0_real = (0.012*X[:, 3] - 0.75*X[:, 5]*X[:, 7] - 0.9*X[:, 4] -
                np.mean(X, axis=1))
    tau_real = X[:, 2] + 0.04*X[:, 9] - 0.35*np.log(X[:, 3])
    prob_of_T = 0.5
    T = np.random.binomial(size=N, n=1, p=prob_of_T)
    normal_errors = np.random.normal(size=[N, ], loc=0.0, scale=1.0)
    Y = mu0_real + tau_real*T + normal_errors

    # Creating training and validation dataset
    X_train, X_valid, T_train, T_valid, Y_train, Y_valid = train_test_split(
        X, T, Y, test_size=0.2, random_state=88)

    # Create a valid instance of InputChecker class
    inp_check = InputChecker(
        [X_train, T_train, Y_train], [X_valid, T_valid, Y_valid],
        [X, T, Y], [30, 20, 15, 10, 5], dropout_rates=None, batch_size=None,
        alpha=0., r_par=0., optimizer='Adam', learning_rate=0.0009,
        max_epochs_without_change=30, max_nepochs=5000, seed=None,
        estimate_ps=False, hidden_layer_sizes_t=None, dropout_rates_t=None,
        batch_size_t=None, alpha_t=0., r_par_t=0., optimizer_t='Adam',
        learning_rate_t=0.0009, max_epochs_without_change_t=30,
        max_nepochs_t=5000, seed_t=None)

    # Add `inp_check` attribute to the class under test
    if request.cls is not None:
        request.cls.inp_check = inp_check

    yield inp_check

    # After all the tests are done, delete the instance that was created
    del inp_check


def assert_message_raised(function, error, message, *args):
    '''Checks that appropriate error is being raised for a given
    function with given parameters and also asserts that the error
    message matches the expected one.'''
    with pytest.raises(error) as error_info:
        function(*args)
    assert str(error_info.value) == message


@pytest.mark.usefixtures('inp_check')
class TestInputChecker():
    '''Test expected exceptions that should be raised with non-valid inputs
    to the causal_net_estimate function.'''

    def test_is_data_list_valid1(self):
        '''Checks if appropriate error is being raised when training,
        validation or test data is not provided as a list.'''
        assert_message_raised(
            self.inp_check._is_data_list_valid, TypeError,
            'Data such as training, validation and test must be provided' +
            ' as lists', tuple(self.inp_check.training_data)
        )

    def test_is_data_list_valid2(self):
        '''Checks if appropriate error is being raised when training,
        validation or test data does not contain all three data arrays.'''
        assert_message_raised(
            self.inp_check._is_data_list_valid, Exception,
            'All data lists must contain array of input features, treatment' +
            ' array and target array', self.inp_check.validation_data[:1]
        )

    def test_is_data_list_valid3(self):
        '''Checks if appropriate error is being raised if features,
        treatment and target arrays are not of the same length in any
        of the data lists.'''
        assert_message_raised(
            self.inp_check._is_data_list_valid, Exception,
            'Length of input features array, treatment array and' +
            ' target array must be equal',
            [self.inp_check.test_data[0], self.inp_check.test_data[1],
             self.inp_check.test_data[2][:-3]]
        )

    def test_is_data_list_valid4(self):
        '''Checks if appropriate error is being raised if treatment and
        target arrays are not provided as arrays of shape Nx1 or N.'''
        assert_message_raised(
            self.inp_check._is_data_list_valid, Exception,
            'Treatment and target arrays must be provided as arrays' +
            ' of shape Nx1 or N',
            [self.inp_check.validation_data[1],
             self.inp_check.validation_data[0],
             self.inp_check.validation_data[2]]
        )

    def test_is_treatment_array_binary(self):
        '''Checks if appropriate error is being raised if treatment array is
        not provided as a binary array of 0s and 1s.'''
        assert_message_raised(
            self.inp_check._is_treatment_array_binary, Exception,
            'Treatment array needs to be a binary array of 0s and 1s',
            np.array([0, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1]))

    @pytest.mark.parametrize(
        'error, message, hidden_layer_sizes',
        [
            (TypeError, 'hidden_layer_sizes must be provided as a list',
             (30, 20, 15, 10, 5)),
            (TypeError, 'hidden_layer_sizes must be provided as a list', None),
            (TypeError, 'Each hidden layer in hidden_layer_sizes list must' +
             ' be provided as an integer', ['50', '30']),
            (ValueError, 'Each hidden layer in hidden_layer_sizes list' +
             ' has to be a positive integer', [30, 20, -15, 10, 5])
        ])
    def test_are_hidden_layer_sizes_valid(self, error, message,
                                          hidden_layer_sizes):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for hidden_layer_sizes is being
        provided.'''
        assert_message_raised(self.inp_check._are_hidden_layer_sizes_valid,
                              error, message, hidden_layer_sizes)

    @pytest.mark.parametrize(
        'error, message, dropout_rates, hidden_layer_sizes',
        [
            (TypeError, 'dropout_rates must be provided as a list or' +
             ' set to None', 'None', [60, 30, 20]),
            (TypeError, 'Dropout rates must be provided as numeric values',
             [0.3, '0.2', 0.], [60, 30, 20]),
            (ValueError, 'Dropout rates for each hidden layer must be a ' +
             'number between 0 and 1', [0.3, 0.2, 2, 0], [30, 20, 15, 10]),
            (Exception, 'dropout_rates list has to be of the same length as' +
             ' the hidden_layer_sizes list', [0.3, 0.2, 0], [30, 20, 15, 5])
        ])
    def test_are_dropout_rates_valid(self, error, message,
                                     dropout_rates, hidden_layer_sizes):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for dropout_rates is being
        provided.'''
        assert_message_raised(self.inp_check._are_dropout_rates_valid, error,
                              message, dropout_rates, hidden_layer_sizes,)

    @pytest.mark.parametrize(
        'error, message, batch_size',
        [
            (TypeError, 'Batch size must be an integer or set to None',
             'None'),
            (ValueError, 'Batch size must be a positive integer', -3),
            (Exception, 'Batch size value should be less then the length of'
             ' the training data', 10000)
        ])
    def test_is_batch_size_valid(self, error, message, batch_size):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for batch_size is being
        provided.'''
        assert_message_raised(self.inp_check._is_batch_size_valid,
                              error, message, batch_size)

    @pytest.mark.parametrize(
        'error, message, alpha',
        [
            (TypeError, 'Alpha parameter must be provided as a float', 0),
            (TypeError, 'Alpha parameter must be provided as a float', None),
            (ValueError, 'Alpha parameter must be a non-negative float', -1.)
        ])
    def test_is_alpha_parameter_valid(self, error, message, alpha):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for alpha parameter is being
        provided.'''
        assert_message_raised(self.inp_check._is_alpha_parameter_valid,
                              error, message, alpha)

    @pytest.mark.parametrize(
        'error, message, r_par',
        [
            (TypeError, 'r_par must be provided as a float', 0),
            (ValueError, 'r_par must be a float between 0 and 1', 1.2),
            (TypeError, 'r_par must be provided as a float', None)
        ])
    def test_is_r_par_valid(self, error, message, r_par):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for r_par is being provided.'''
        assert_message_raised(self.inp_check._is_r_par_valid,
                              error, message, r_par)

    @pytest.mark.parametrize(
        'error, message, optimizer',
        [
            (TypeError, 'Optimizer must be provided as a string', None),
            (TypeError, 'Optimizer must be provided as a string', ['Adam']),
            (ValueError, "Optimizer not recognized. It must be one of" +
             " the three options: 'Adam', 'GradientDescent', or 'RMSprop'",
             'Adm'),
            (ValueError, "Optimizer not recognized. It must be one of" +
             " the three options: 'Adam', 'GradientDescent', or 'RMSprop'",
             'rmsprop')
        ])
    def test_is_optimizer_valid(self, error, message, optimizer):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for optimizer parameter is
        being provided.'''
        assert_message_raised(self.inp_check._is_optimizer_valid,
                              error, message, optimizer)

    @pytest.mark.parametrize(
        'error, message, learning_rate',
        [
            (TypeError, 'Learning rate must be provided as a float', None),
            (ValueError, 'Learning rate must be a positive float', 0.),
            (ValueError, 'Learning rate must be a positive float', -0.1)
        ])
    def test_is_learning_rate_valid(self, error, message, learning_rate):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for learning rate is being provided.'''
        assert_message_raised(self.inp_check._is_learning_rate_valid,
                              error, message, learning_rate)

    @pytest.mark.parametrize(
        'error, message, max_epochs',
        [
            (TypeError, 'max_nepochs must be provided as an integer', None),
            (TypeError, 'max_nepochs must be provided as an integer', '100'),
            (ValueError, 'max_nepochs must be a positive integer', 0)
        ])
    def test_is_max_epochs_valid(self, error, message, max_epochs):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for max_epochs is being provided.'''
        assert_message_raised(self.inp_check._is_max_epochs_valid,
                              error, message, max_epochs)

    @pytest.mark.parametrize(
        'error, message, max_epochs_without_change',
        [
            (TypeError, 'max_epochs_without_change must be provided as' +
             ' an integer', None),
            (TypeError, 'max_epochs_without_change must be provided as' +
             ' an integer', 10.),
            (ValueError, 'max_epochs_without_change must be a' +
             ' non-negative integer', -2)
        ])
    def test_is_max_epochs_without_change_valid(self, error, message,
                                                max_epochs_without_change):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for max_epochs_without_change
        is being provided.'''
        assert_message_raised(
            self.inp_check._is_max_epochs_without_change_valid,
            error, message, max_epochs_without_change)

    @pytest.mark.parametrize(
        'error, message, seed',
        [
            (TypeError, 'Seed must be provided as an integer or set to None',
             'None'),
            (TypeError, 'Seed must be provided as an integer or set to None',
             3.),
            (ValueError, 'Seed must be a positive integer', -25)
        ])
    def test_is_seed_par_valid(self, error, message, seed):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for seed is being provided.'''
        assert_message_raised(self.inp_check._is_seed_par_valid,
                              error, message, seed)

    @pytest.mark.parametrize(
        'error, message, estimate_ps',
        [
            (TypeError, 'estimate_ps must be provided as a boolean', 'False'),
            (TypeError, 'estimate_ps must be provided as a boolean', None)
        ])
    def test_is_estimate_ps_valid(self, error, message, estimate_ps):
        '''Checks if appropriate type of error and error message are
        being raised when invalid input for estimate_ps parameter is
        being provided.'''
        assert_message_raised(self.inp_check._is_estimate_ps_valid,
                              error, message, estimate_ps)

    def test_no_error_is_being_raised1(self):
        '''Checks that no error is raised when valid inputs are provided'''
        try:
            self.inp_check.check_all_parameters()

            # Enable estimation of propensity scores and check again that no
            # error is being raised
            self.inp_check.estimate_ps = True
            self.inp_check.hidden_layer_sizes_t = [60, 30]
            self.inp_check.check_all_parameters()
        except Exception as e:
            raise pytest.fail("DID RAISE {0}".format(e))

    def test_no_error_is_being_raised2(self):
        '''Checks that no error is raised when invalid input is
        provided for the propensity score neural network parameters in
        case when estimate_ps is set to False.'''
        try:
            self.inp_check.estimate_ps = False
            self.inp_check.hidden_layer_sizes_t = ['60', '30']
            self.inp_check.dropout_rates_t = [0.2, 0., 0.]

            self.inp_check.check_all_parameters()

        except Exception as e:
            raise pytest.fail("DID RAISE {0}".format(e))


if __name__ == '__main__':
    test_input_checker()
