
import numpy as np

import theano

import theano.tensor as T

import network_layer

class Network:

    def __init__(self, m_w_init, v_w_init, a_init, b_init):

        # We create the different layers

        self.layers = []

        if len(m_w_init) > 1:
            for m_w, v_w in zip(m_w_init[ : -1 ], v_w_init[ : -1 ]):
                self.layers.append(network_layer.Network_layer(m_w, v_w, True))

        self.layers.append(network_layer.Network_layer(m_w_init[ -1 ],
            v_w_init[ -1 ], False))

        # We create mean and variance parameters from all layers

        self.params_m_w = []
        self.params_v_w = []
        self.params_w = []
        for layer in self.layers:
            self.params_m_w.append(layer.m_w)
            self.params_v_w.append(layer.v_w)
            self.params_w.append(layer.w)

        # We create the theano variables for a and b

        self.a = theano.shared(float(a_init))
        self.b = theano.shared(float(b_init))

    def output_deterministic(self, x):

        # Recursively compute output

        for layer in self.layers:
            x = layer.output_deterministic(x)

        return x

    def output_probabilistic(self, m):

        v = T.zeros_like(m)

        # Recursively compute output

        for layer in self.layers:
            m, v = layer.output_probabilistic(m, v)

        return (m[ 0 ], v[ 0 ])

    def logZ_Z1_Z2(self, x, y):

        m, v = self.output_probabilistic(x)

        v_final = v + self.b / (self.a - 1)
        v_final1 = v + self.b / self.a
        v_final2 = v + self.b / (self.a + 1)

        logZ = -0.5 * (T.log(v_final) + (y - m)**2 / v_final)
        logZ1 = -0.5 * (T.log(v_final1) + (y - m)**2 / v_final1)
        logZ2 = -0.5 * (T.log(v_final2) + (y - m)**2 / v_final2)

        return (logZ, logZ1, logZ2)

    def generate_updates(self, logZ, logZ1, logZ2):

        updates = []
        for i in range(len(self.params_m_w)):
            updates.append((self.params_m_w[ i ], self.params_m_w[ i ] + \
                self.params_v_w[ i ] * T.grad(logZ, self.params_m_w[ i ])))
            updates.append((self.params_v_w[ i ], self.params_v_w[ i ] - \
               self.params_v_w[ i ]**2 * \
                (T.grad(logZ, self.params_m_w[ i ])**2 - 2 * \
                T.grad(logZ, self.params_v_w[ i ]))))

        updates.append((self.a, 1.0 / (T.exp(logZ2 - 2 * logZ1 + logZ) * \
            (self.a + 1) / self.a - 1.0)))
        updates.append((self.b, 1.0 / (T.exp(logZ2 - logZ1) * (self.a + 1) / \
            (self.b) - T.exp(logZ1 - logZ) * self.a / self.b)))

        return updates

    def get_params(self):

        m_w = []
        v_w = []
        for layer in self.layers:
            m_w.append(layer.m_w.get_value())
            v_w.append(layer.v_w.get_value())

        return { 'm_w': m_w, 'v_w': v_w , 'a': self.a.get_value(),
            'b': self.b.get_value() }

    def set_params(self, params):

        for i in range(len(self.layers)):
            self.layers[ i ].m_w.set_value(params[ 'm_w' ][ i ])
            self.layers[ i ].v_w.set_value(params[ 'v_w' ][ i ])

        self.a.set_value(params[ 'a' ])
        self.b.set_value(params[ 'b' ])

    def remove_invalid_updates(self, new_params, old_params):

        m_w_new = new_params[ 'm_w' ]
        v_w_new = new_params[ 'v_w' ]
        m_w_old = old_params[ 'm_w' ]
        v_w_old = old_params[ 'v_w' ]

        for i in range(len(self.layers)):
            index1 = np.where(v_w_new[ i ] <= 1e-100)
            index2 = np.where(np.logical_or(np.isnan(m_w_new[ i ]),
                np.isnan(v_w_new[ i ])))

            index = [ np.concatenate((index1[ 0 ], index2[ 0 ])),
                np.concatenate((index1[ 1 ], index2[ 1 ])) ]

            if len(index[ 0 ]) > 0:
                m_w_new[ i ][ index ] = m_w_old[ i ][ index ]
                v_w_new[ i ][ index ] = v_w_old[ i ][ index ]

    def sample_w(self):

        w = []
        for i in range(len(self.layers)):
            w.append(self.params_m_w[ i ].get_value() + \
                np.random.randn(self.params_m_w[ i ].get_value().shape[ 0 ], \
                self.params_m_w[ i ].get_value().shape[ 1 ]) * \
                np.sqrt(self.params_v_w[ i ].get_value()))

        for i in range(len(self.layers)):
            self.params_w[ i ].set_value(w[ i ])
