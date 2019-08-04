
import sys

import math

import numpy as np

import theano

import theano.tensor as T

import network

import prior

class PBP:

    def __init__(self, layer_sizes, mean_y_train, std_y_train):

        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # We initialize the prior

        self.prior = prior.Prior(layer_sizes, var_targets)

        # We create the network

        params = self.prior.get_initial_params()
        self.network = network.Network(params[ 'm_w' ], params[ 'v_w' ],
            params[ 'a' ], params[ 'b' ])

        # We create the input and output variables in theano

        self.x = T.vector('x')
        self.y = T.scalar('y')
        
        # A function for computing the value of logZ, logZ1 and logZ2

        self.logZ, self.logZ1, self.logZ2 = \
            self.network.logZ_Z1_Z2(self.x, self.y)

        # We create a theano function for updating the posterior

        self.adf_update = theano.function([ self.x, self.y ], self.logZ,
            updates = self.network.generate_updates(self.logZ, self.logZ1,
            self.logZ2))

        # We greate a theano function for the network predictive distribution

        self.predict_probabilistic = theano.function([ self.x ],
            self.network.output_probabilistic(self.x))

        self.predict_deterministic = theano.function([ self.x ],
            self.network.output_deterministic(self.x))

    def do_pbp(self, X_train, y_train, n_iterations):

        if n_iterations > 0:

            # We first do a single pass

            self.do_first_pass(X_train, y_train)

            # We refine the prior

            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

            sys.stdout.write('{}\n'.format(0))
            sys.stdout.flush()

            for i in range(int(n_iterations) - 1):

                # We do one more pass

                params = self.prior.get_params()
                self.do_first_pass(X_train, y_train)

                # We refine the prior

                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)

                sys.stdout.write('{}\n'.format(i + 1))
                sys.stdout.flush()

    def get_deterministic_output(self, X_test):

        output = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            output[ i ] = self.predict_deterministic(X_test[ i, : ])
            output[ i ] = output[ i ] * self.std_y_train + self.mean_y_train

        return output

    def get_predictive_mean_and_variance(self, X_test):

        mean = np.zeros(X_test.shape[ 0 ])
        variance = np.zeros(X_test.shape[ 0 ])
        for i in range(X_test.shape[ 0 ]):
            m, v = self.predict_probabilistic(X_test[ i, : ])
            m = m * self.std_y_train + self.mean_y_train
            v = v * self.std_y_train**2
            mean[ i ] = m
            variance[ i ] = v

        v_noise = self.network.b.get_value() / \
            (self.network.a.get_value() - 1) * self.std_y_train**2

        return mean, variance, v_noise

    def do_first_pass(self, X, y):

        permutation = np.random.choice(range(X.shape[ 0 ]), X.shape[ 0 ],
            replace = False)

        counter = 0
        for i in permutation:

            old_params = self.network.get_params()
            logZ = self.adf_update(X[ i, : ], y[ i ])
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)

            if counter % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

            counter += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def sample_w(self):

        self.network.sample_w()
