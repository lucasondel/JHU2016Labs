
"""Gaussian distribution."""

import numpy as np
from scipy.misc import logsumexp
from scipy.special import gamma

class Gaussian(object):

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def pdf(self, x):
        norm = np.sqrt(2 * np.pi * self.var)
        return np.exp((-.5 / self.var) * ((x - self.mean) ** 2)) / norm

    def logLikelihood(self, x):
        return np.sum(np.log(self.pdf(x)))

    @staticmethod
    def maximumLikelihood(X):
        N = len(X)
        mean = np.sum(X)/N
        var = np.sum(X**2)/N
        return Gaussian(mean, var)


class GMM(object):

    def __init__(self, means, variances, weights):
        self.weights = weights
        self.gaussians = [Gaussian(means[k], variances[k]) for k in
                          range(len(means))]

        assert np.isclose(np.sum(self.weights), 1.), 'The weights should sum up to one.'

    @property
    def k(self):
        return len(self.gaussians)

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise KeyError('The index should be an integer.')
        if key >= self.k or key < 0:
            raise IndexError('Index out of bounds.')
        return self.gaussians[key]

    def sampleData(self, n=1000):
        X = []  # all the data
        for k in range(self.k):
            gaussian = self.gaussians[k]
            Xk = np.random.normal(gaussian.mean, np.sqrt(gaussian.var),
                                  int(n * self.weights[k]))
            X.append(Xk)
        X = np.hstack(X)

        np.random.shuffle(X)
        return X

    def pdf(self, x, sum_pdf=True):
        retval = np.zeros((len(x), self.k))
        for k in range(self.k):
            retval[:,k] = self.weights[k] * self.gaussians[k].pdf(x)
        if sum_pdf:
            retval = retval.sum(axis=1)
        return retval

    def logLikelihood(self, X):
        return np.sum(np.log(self.pdf(X)))

    def logLikelihoodPerComponent(self, X):
        return np.log(self.pdf(X, sum_pdf=False))

    def EStep(self, X):
        llh_per_comp = self.logLikelihoodPerComponent(X)
        norms = logsumexp(llh_per_comp, axis=1)
        resps = np.exp(llh_per_comp.T - norms).T

        return resps

    def MStep(self, X, resps):
        for k in range(self.k):
            mean = np.sum(resps[:, k] * X) / resps[:, k].sum()
            self.gaussians[k].mean = mean
            self.gaussians[k].var = \
                    np.sum((resps[:, k] * ((X - mean) ** 2))) / resps[:, k].sum()
            self.weights[k] = np.sum(resps[:, k]) / X.shape[0]

    def EM(self, X, threshold=1e-2):
        previous_llh = float('-inf')
        current_llh = self.logLikelihood(X)
        while current_llh - previous_llh > threshold:
            self._EMStep(X)
            previous_llh = current_llh
            current_llh = self.logLikelihood(X)


class Dirichlet(object):

    def __init__(self, alphas):
        self.alphas = alphas

    def sample(self):
        return np.random.dirichlet(self.alphas)

    def posterior(self, x):
        Nk = x.sum(axis=0)
        return Dirichlet(self.alphas + Nk)


class BayesianGMM(object):

    def __init__(self, alphas, m, kappa, a, b):
        self.dir0 = Dirichlet(alphas)
        self.NG0 = NormalGamma(m, kappa, a, b)

        weights = self.dir0.sample()
        means = []
        variances = []
        for k in range(len(weights)):
            m, p = self.NG0.sample()
            means.append(m)
            variances.append(1/p)
        self.gmm = GMM(means, variances, weights)

        self.count_mv = 1
        self.count_w = 1
        self.means_avg = np.array(means)
        self.variances_avg = np.array(variances)
        self.weights_avg = np.array(weights)

    def sampleLatentVariables(self, x):
        llh_per_comp = self.gmm.logLikelihoodPerComponent(x)
        norms = logsumexp(llh_per_comp, axis=1)
        resps = np.exp(llh_per_comp.T - norms).T
        zs = []
        for i in range(len(resps)):
            zs.append(np.random.multinomial(1, resps[i]))
        return np.array(zs, dtype=int)

    def sampleMeansVariances(self, x, zs):
        for k in range(self.gmm.k):
            indices = np.where(zs[:, k] == 1)
            if len(indices[0]) > 0:
                NG = self.NG0.posterior(x[indices])
            else:
                NG = self.NG0
            mean, prec = NG.sample()
            self.gmm.gaussians[k].mean = mean
            self.gmm.gaussians[k].var = 1 / prec
        self.means_avg += np.array([g.mean for g in self.gmm.gaussians])
        self.variances_avg += np.array([g.var for g in self.gmm.gaussians])
        self.count_mv += 1

    def sampleWeights(self, zs):
        dirichlet = self.dir0.posterior(zs)
        self.gmm.weights = dirichlet.sample()
        self.weights_avg += self.gmm.weights
        self.count_w += 1

    def averageGMM(self):
        means = self.means_avg / self.count_mv
        variances = self.variances_avg / self.count_mv
        weights = self.weights_avg /  self.count_w
        return GMM(means, variances, weights)


class NormalGamma(object):

    def __init__(self, mean, kappa, a, b):
        self.mean = mean
        self.kappa = kappa
        self.a = a
        self.b = b

    def sample(self):
        prec = np.random.gamma(self.a, 1/self.b)
        mu = np.random.normal(self.mean, np.sqrt(1/(self.kappa*prec)))
        return mu, prec

    def _normalPdf(self, x, y):
        var = 1/(self.kappa * y)
        norm = np.sqrt(2 * np.pi * var)
        return np.exp((-.5 / var) * ((x - self.mean) ** 2)) / norm

    def _gammaPdf(self, y):
        norm = (self.b**self.a)/gamma(self.a)
        return y**(self.a - 1) * np.exp(-self.b*y) * norm

    def pdf(self, x, y):
        assert len(x) == len(y), 'x and y should have the same dimension.'

        return self._normalPdf(x, y) * self._gammaPdf(y)

    def posterior(self, x):
        N = len(x)
        x_bar = x.mean()
        m = (self.kappa * self.mean + N * x_bar)/(self.kappa + N)
        kappa = self.kappa + N
        a = self.a + .5* N
        b = self.b + 0.5 * np.sum((x - x_bar)**2)
        b += (self.kappa*N*((x_bar - self.mean)**2)) / (2*(self.kappa + N))
        return NormalGamma(m, kappa, a, b)

    def predictiveDensity(self):
        nu = 2*self.a
        gamma = (self.a*self.kappa) /(self.b*(self.kappa + 1))
        return StudentT(self.mean, nu, gamma)


class StudentT(object):

    def __init__(self, mean, nu, gamma):
        self.mean = mean
        self.nu = nu
        self.gamma = gamma

    def pdf(self, x):
        norm = np.sqrt(self.gamma/(np.pi*self.nu))
        norm *= gamma(.5*self.nu + .5) / gamma(0.5*self.nu)
        density = 1 + (self.gamma*(x-self.mean)**2)/self.nu
        density = density**(-.5*self.nu - .5)
        return norm * density

    def logLikelihood(self, x):
        return np.sum(np.log(self.pdf(x)))

