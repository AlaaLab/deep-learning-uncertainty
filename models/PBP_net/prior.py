
import numpy as np

class Prior:

    def __init__(self, layer_sizes, var_targets):

        # We refine the factor for the prior variance on the weights

        n_samples = 3.0
        v_observed = 1.0
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed

        # We refine the factor for the prior variance on the noise

        n_samples = 3.0
        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * var_targets

        self.a_sigma_hat_nat = a_sigma - 1
        self.b_sigma_hat_nat = -b_sigma

        # We refine the gaussian prior on the weights

        self.rnd_m_w = []
        self.m_w_hat_nat = []
        self.v_w_hat_nat = []
        self.a_w_hat_nat = []
        self.b_w_hat_nat = []
        for size_out, size_in in zip(layer_sizes[ 1 : ], layer_sizes[ : -1 ]):
            self.rnd_m_w.append(1.0 / np.sqrt(size_in + 1) *
                np.random.randn(size_out, size_in + 1))
            self.m_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            self.v_w_hat_nat.append((self.a_w - 1) / self.b_w * \
                np.ones((size_out, size_in + 1)))
            self.a_w_hat_nat.append(np.zeros((size_out, size_in + 1)))
            self.b_w_hat_nat.append(np.zeros((size_out, size_in + 1)))

    def get_initial_params(self):

        m_w = []
        v_w = []
        for i in range(len(self.rnd_m_w)):
            m_w.append(self.rnd_m_w[ i ])
            v_w.append(1.0 / self.v_w_hat_nat[ i ])

        return { 'm_w': m_w, 'v_w': v_w , 'a': self.a_sigma_hat_nat + 1,
            'b': -self.b_sigma_hat_nat }

    def get_params(self):

        m_w = []
        v_w = []
        for i in range(len(self.rnd_m_w)):
            m_w.append(self.m_w_hat_nat[ i ] / self.v_w_hat_nat[ i ])
            v_w.append(1.0 / self.v_w_hat_nat[ i ])

        return { 'm_w': m_w, 'v_w': v_w , 'a': self.a_sigma_hat_nat + 1,
            'b': -self.b_sigma_hat_nat }

    def refine_prior(self, params):

        for i in range(len(params[ 'm_w' ])):
            for j in range(params[ 'm_w' ][ i ].shape[ 0 ]):
                for k in range(params[ 'm_w' ][ i ].shape[ 1 ]):

                    # We obtain the parameters of the cavity distribution

                    v_w_nat = 1.0 / params[ 'v_w' ][ i ][ j, k ]
                    m_w_nat = params[ 'm_w' ][ i ][ j, k ] / \
                        params[ 'v_w' ][ i ][ j, k ]
                    v_w_cav_nat = v_w_nat - self.v_w_hat_nat[ i ][ j, k ]
                    m_w_cav_nat = m_w_nat - self.m_w_hat_nat[ i ][ j, k ]
                    v_w_cav = 1.0 / v_w_cav_nat
                    m_w_cav = m_w_cav_nat / v_w_cav_nat
                    a_w_nat = self.a_w - 1
                    b_w_nat = -self.b_w
                    a_w_cav_nat = a_w_nat - self.a_w_hat_nat[ i ][ j, k ]
                    b_w_cav_nat = b_w_nat - self.b_w_hat_nat[ i ][ j, k ]
                    a_w_cav = a_w_cav_nat + 1
                    b_w_cav = -b_w_cav_nat

                    if v_w_cav > 0 and b_w_cav > 0 and a_w_cav > 1 and \
                        v_w_cav < 1e6:

                        # We obtain the values of the new parameters of the
                        # posterior approximation

                        v = v_w_cav + b_w_cav / (a_w_cav - 1)
                        v1  = v_w_cav + b_w_cav / a_w_cav
                        v2  = v_w_cav + b_w_cav / (a_w_cav + 1)
                        logZ = -0.5 * np.log(v) - 0.5 * m_w_cav**2 / v
                        logZ1 = -0.5 * np.log(v1) - 0.5 * m_w_cav**2 / v1
                        logZ2 = -0.5 * np.log(v2) - 0.5 * m_w_cav**2 / v2
                        d_logZ_d_m_w_cav = -m_w_cav / v
                        d_logZ_d_v_w_cav = -0.5 / v + 0.5 * m_w_cav**2 / v**2
                        m_w_new = m_w_cav + v_w_cav * d_logZ_d_m_w_cav
                        v_w_new = v_w_cav - v_w_cav**2 * \
                            (d_logZ_d_m_w_cav**2 - 2 * d_logZ_d_v_w_cav)
                        a_w_new = 1.0 / (np.exp(logZ2 - 2 * logZ1 + logZ) * \
                            (a_w_cav + 1) / a_w_cav - 1.0)
                        b_w_new = 1.0 / (np.exp(logZ2 - logZ1) * \
                            (a_w_cav + 1) / (b_w_cav) - np.exp(logZ1 - \
                            logZ) * a_w_cav / b_w_cav)
                        v_w_new_nat = 1.0 / v_w_new
                        m_w_new_nat = m_w_new / v_w_new
                        a_w_new_nat = a_w_new - 1
                        b_w_new_nat = -b_w_new

                        # We update the parameters of the approximate factor,
                        # whih is given by the ratio of the new posterior
                        # approximation and the cavity distribution

                        self.m_w_hat_nat[ i ][ j, k ] = m_w_new_nat - \
                            m_w_cav_nat
                        self.v_w_hat_nat[ i ][ j, k ] = v_w_new_nat - \
                            v_w_cav_nat
                        self.a_w_hat_nat[ i ][ j, k ] = a_w_new_nat - \
                            a_w_cav_nat
                        self.b_w_hat_nat[ i ][ j, k ] = b_w_new_nat - \
                            b_w_cav_nat

                        # We update the posterior approximation

                        params[ 'm_w' ][ i ][ j, k ] = m_w_new
                        params[ 'v_w' ][ i ][ j, k ] = v_w_new

                        self.a_w = a_w_new
                        self.b_w = b_w_new

        return params
