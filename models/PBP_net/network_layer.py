
import math

import theano

import theano.tensor as T

class Network_layer:

    def __init__(self, m_w_init, v_w_init, non_linear = True):

        # We create the theano variables for the means and variances

        self.m_w = theano.shared(value = m_w_init.astype(theano.config.floatX),
            name='m_w', borrow = True)
        self.v_w = theano.shared(value = v_w_init.astype(theano.config.floatX),
            name='v_w', borrow = True)
        self.w = theano.shared(value = m_w_init.astype(theano.config.floatX),
            name='w', borrow = True)

        # We store the type of activation function

        self.non_linear = non_linear

        # We store the number of inputs

        self.n_inputs = theano.shared(float(m_w_init.shape[ 1 ]))

    @staticmethod
    def n_pdf(x):

        return 1.0 / T.sqrt(2 * math.pi) * T.exp(-0.5 * x**2)

    @staticmethod
    def n_cdf(x):

        return 0.5 * (1.0 + T.erf(x / T.sqrt(2.0)))

    @staticmethod
    def gamma(x):

        return Network_layer.n_pdf(x) / Network_layer.n_cdf(-x)

    @staticmethod
    def beta(x):

        return Network_layer.gamma(x) * (Network_layer.gamma(x) - x)

    def output_probabilistic(self, m_w_previous, v_w_previous):

        # We add an additional deterministic input with mean 1 and variance 0

        m_w_previous_with_bias = \
            T.concatenate([ m_w_previous, T.alloc(1, 1) ], 0)
        v_w_previous_with_bias = \
            T.concatenate([ v_w_previous, T.alloc(0, 1) ], 0)

        # We compute the mean and variance after the linear operation

        m_linear = T.dot(self.m_w, m_w_previous_with_bias) / T.sqrt(self.n_inputs)
        v_linear = (T.dot(self.v_w, v_w_previous_with_bias) + \
            T.dot(self.m_w**2, v_w_previous_with_bias) + \
            T.dot(self.v_w, m_w_previous_with_bias**2)) / self.n_inputs

        if (self.non_linear):

            # We compute the mean and variance after the ReLU activation

            alpha = m_linear / T.sqrt(v_linear)
            gamma = Network_layer.gamma(-alpha)
            gamma_robust = -alpha - 1.0 / alpha + 2.0 / alpha**3
            gamma_final = T.switch(T.lt(-alpha, T.fill(alpha, 30)), gamma, gamma_robust)

            v_aux = m_linear + T.sqrt(v_linear) * gamma_final

            m_a = Network_layer.n_cdf(alpha) * v_aux
            v_a = m_a * v_aux * Network_layer.n_cdf(-alpha) + \
                Network_layer.n_cdf(alpha) * v_linear * \
                (1 - gamma_final * (gamma_final + alpha))

            return (m_a, v_a)

        else:

            return (m_linear, v_linear)

    def output_deterministic(self, output_previous):

        # We add an additional input with value 1

        output_previous_with_bias = \
            T.concatenate([ output_previous, T.alloc(1, 1) ], 0) / \
            T.sqrt(self.n_inputs)

        # We compute the mean and variance after the linear operation

        a = T.dot(self.w, output_previous_with_bias)

        if (self.non_linear):

            # We compute the ReLU activation

            a = T.switch(T.lt(a, T.fill(a, 0)), T.fill(a, 0), a)

        return a
