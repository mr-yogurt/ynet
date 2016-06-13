__author__ = 'mr_yogurt'

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#it's entirely possible this code doesn't work correctly considering i haven't tested it recently. it worked when i was using it but that's no guarantee

np.random.seed(0)
srng = RandomStreams(seed=1)
# for some reason the seed can't be zero so one is the next best choice


class FeedforwardNet:
    W = []
    b = []
    o = []

    #borrowed_params is another net that you want to use the weights and biases from without copying
    #params is the npparams() of another net
    #no error handling whatsoever since this was meant for personal use (same reason for very little documentation)
    def __init__(self, X, topology, activations, borrowed_params=None, params=None, wis=1.0):
        self.o.append(activations[0](X))
        activations = activations[1:]
        self.adam_w_m = []
        self.adam_w_v = []
        self.adam_b_m = []
        self.adam_b_v = []
        if borrowed_params is None and params is None:
            for wl, nwl, f in zip(topology[:-1], topology[1:], activations):
		# xavier initialization of weights
                self.W.append(theano.shared(np.array(np.random.normal(
                    loc=0.0, scale=(.5 * wis * (wl+nwl))**-.5, size=(wl, nwl)), dtype=theano.config.floatX),
                    name='w', borrow=True))
                self.b.append(theano.shared(np.array(np.random.standard_normal(nwl), dtype=theano.config.floatX),
                                            name='b', borrow=True))
                self.o.append(f(T.dot(self.o[-1], self.W[-1]) + self.b[-1]))

                # initialization of adam optimizer variables. i realize this should probably be done lazily when the adam update method gets called but whatever
                self.adam_w_m.append(theano.shared(np.zeros((wl, nwl), dtype=theano.config.floatX),
                                                   name='adam_w_m', borrow=True))
                self.adam_w_v.append(theano.shared(np.zeros((wl, nwl), dtype=theano.config.floatX),
                                                   name='adam_w_v', borrow=True))
                self.adam_b_m.append(theano.shared(np.zeros(nwl, dtype=theano.config.floatX),
                                                   name='adam_b_m', borrow=True))
                self.adam_b_v.append(theano.shared(np.zeros(nwl, dtype=theano.config.floatX),
                                                   name='adam_b_v', borrow=True))

            self.stepnum = theano.shared(np.array(1.0, dtype=theano.config.floatX), borrow=True, name='adam_step')

            self.X = X
            self.Y = self.o[-1]
            self.compute = theano.function(inputs=[self.X], outputs=self.Y)
        elif params is None:
            for wl, nwl, f, ik in zip(topology[:-1], topology[1:], activations, range(0, len(topology)-1)):
                if ik < len(borrowed_params[0]):
                    self.W.append(borrowed_params[0][ik])
                    self.b.append(borrowed_params[1][ik])
                else:
                    self.W.append(
                        theano.shared(
                            np.array(np.random.normal(
                                loc=0.0,
                                scale=(.5 * wis * (wl+nwl))**-.5, size=(wl, nwl)),
                            dtype=theano.config.floatX),
                        name='w', borrow=True))
                    self.b.append(
                        theano.shared(
                            np.array(
                                np.random.standard_normal(nwl),
                                dtype=theano.config.floatX),
                            name='b', borrow=True))
                self.o.append(f(T.dot(self.o[-1], self.W[-1]) + self.b[-1]))

            self.stepnum = theano.shared(np.array(1.0, dtype=theano.config.floatX), borrow=True, name='adam_step')
            self.X = X
            self.Y = self.o[-1]
            self.compute = theano.function(inputs=[self.X], outputs=self.Y)
        else:
            for wl, nwl, f, ik in zip(topology[:-1], topology[1:], activations, range(0, len(topology)-1)):
                if ik < len(params[0]):
                    self.W.append(theano.shared(params[0][ik], borrow=True))
                    self.b.append(theano.shared(params[1][ik], borrow=True))
                else:
                    self.W.append(
                        theano.shared(
                            np.array(np.random.normal(
                                loc=0.0,
                                scale=(.5 * wis * (wl+nwl))**-.5, size=(wl, nwl)),
                            dtype=theano.config.floatX),
                        name='w', borrow=True))
                    self.b.append(
                        theano.shared(
                            np.array(
                                np.random.standard_normal(nwl),
                                dtype=theano.config.floatX),
                            name='b', borrow=True))
                self.o.append(f(T.dot(self.o[-1], self.W[-1]) + self.b[-1]))
                self.adam_w_m.append(theano.shared(np.zeros((wl, nwl), dtype=theano.config.floatX),
                                                   name='adam_w_m', borrow=True))
                self.adam_w_v.append(theano.shared(np.zeros((wl, nwl), dtype=theano.config.floatX),
                                                   name='adam_w_v', borrow=True))
                self.adam_b_m.append(theano.shared(np.zeros(nwl, dtype=theano.config.floatX),
                                                   name='adam_b_m', borrow=True))
                self.adam_b_v.append(theano.shared(np.zeros(nwl, dtype=theano.config.floatX),
                                                   name='adam_b_v', borrow=True))
            self.stepnum = theano.shared(np.array(1.0, dtype=theano.config.floatX), borrow=True, name='adam_step')
            self.X = X
            self.Y = self.o[-1]
            self.compute = theano.function(inputs=[self.X], outputs=self.Y)

    def useweights(self, net):
        for i in range(0, len(self.W)):
            self.W[i] = net.W[i]  # i realize that standard python procedure is not to use range(...) for iteration but i'm lazy
        for i in range(0, len(self.b)):
            self.b[i] = net.b[i]

    def gdupdate(self, cost, lr):
        updates = []
        for weights, biases in zip(self.W, self.b):
            updates.append((weights, weights - lr * T.grad(cost=cost, wrt=weights)))
            updates.append((biases, biases - lr * T.grad(cost=cost, wrt=biases)))
        return updates

    def adamupdate(self, cost, lr=.001, b1=.9, b2=.999):
        updates = []
        for weights, m, v, biases, mb, vb in zip(self.W, self.adam_w_m, self.adam_w_v,
                                                 self.b, self.adam_b_m, self.adam_b_v):
            m_u = (b1 * m) + ((1.0 - b1) * T.grad(cost=cost, wrt=weights))
            v_u = (b2 * v) + ((1.0 - b2) * T.power(T.grad(cost=cost, wrt=weights), 2.0))
            m_u_c = m_u / (1.0 - T.power(b1, self.stepnum))
            v_u_c = v_u / (1.0 - T.power(b2, self.stepnum))
            u = lr * m_u_c / (1e-7 + T.power(v_u_c, .5))

            updates.append((m, m_u))
            updates.append((v, v_u))
            updates.append((weights, weights - u))

            m_ub = (b1 * mb) + ((1.0 - b1) * T.grad(cost=cost, wrt=biases))
            v_ub = (b2 * vb) + ((1.0 - b2) * T.power(T.grad(cost=cost, wrt=biases), 2.0))
            m_u_cb = m_ub / (1.0 - T.power(b1, self.stepnum))
            v_u_cb = v_ub / (1.0 - T.power(b2, self.stepnum))

            ub = lr * m_u_cb / (1e-7 + T.power(v_u_cb, .5))
            updates.append((mb, m_ub))
            updates.append((vb, v_ub))
            updates.append((biases, biases - ub))
        return updates + [(self.stepnum, self.stepnum + 1.0)]
    
    #probably should move these error functions out of the class. oh well.
    def cross_entropy(self, targets):
        return -T.mean(targets * T.log(1e-10 + self.Y))

    #hackish method for dealing with a dataset that does not have an equal distribution between the two classes
    def biased_binary_cross_entropy(self, target, bias):
        return -T.mean((1 + (target * bias)) * T.log(1e-8 + T.abs_(1 - target - self.Y)))

    def mean_square(self, targets):
        return T.mean(T.power(self.Y - targets, 2.0))

    def root_mean_square(self, targets):
        return T.power(T.mean(T.power(self.Y - targets, 2.0)), 0.5)

    def mean_abs_error(self, targets):
        return T.mean(T.abs_(targets - self.Y))

    def classification_error(self, targets):
        return T.mean(T.abs_(T.round(self.Y) - targets))

    def exp_square_error(self, targets):
        return T.mean(T.exp((targets-self.Y)**2) - 1)

    def npparams(self):
        return [k.get_value() for k in self.W], [k.get_value() for k in self.b]
