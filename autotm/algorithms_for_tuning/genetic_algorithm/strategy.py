# strategies of automatic EA configuration

# CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

from math import sqrt, log

# code source: https://github.com/DEAP/deap/blob/master/deap/cma.py
import numpy


class Strategy(object):
    def __init__(self, centroid, sigma, **kwargs):
        self.params = kwargs

        # Create a centroid as a numpy array
        self.centroid = numpy.array(centroid)

        self.dim = len(self.centroid)
        self.sigma = sigma
        self.pc = numpy.zeros(self.dim)
        self.ps = numpy.zeros(self.dim)
        self.chiN = sqrt(self.dim) * (
            1 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim**2)
        )

        self.C = self.params.get("cmatrix", numpy.identity(self.dim))
        self.diagD, self.B = numpy.linalg.eigh(self.C)

        indx = numpy.argsort(self.diagD)
        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.lambda_ = self.params.get("lambda_", int(4 + 3 * log(self.dim)))
        self.update_count = 0
        self.computeParams(self.params)

    def generate(self, ind_init):
        r"""Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.
        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals.
        """
        arz = numpy.random.standard_normal((self.lambda_, self.dim))
        arz = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)
        return [ind_init(a) for a in arz]

    def update(self, population):
        """Update the current covariance matrix strategy from the
        *population*.
        :param population: A list of individuals from which to update the
                           parameters.
        """
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        old_centroid = self.centroid
        self.centroid = numpy.dot(self.weights, population[0: self.mu])

        c_diff = self.centroid - old_centroid

        # Cumulation : update evolution path
        self.ps = (1 - self.cs) * self.ps + sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) / self.sigma * numpy.dot(
            self.B, (1.0 / self.diagD) * numpy.dot(self.B.T, c_diff)
        )

        hsig = float(
            (
                numpy.linalg.norm(self.ps)
                / sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.update_count + 1.0)))
                / self.chiN
                < (1.4 + 2.0 / (self.dim + 1.0))
            )
        )

        self.update_count += 1

        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) / self.sigma * c_diff

        # Update covariance matrix
        artmp = population[0: self.mu] - old_centroid
        self.C = (
            (
                1
                - self.ccov1
                - self.ccovmu
                + (1 - hsig) * self.ccov1 * self.cc * (2 - self.cc)
            )
            * self.C
            + self.ccov1 * numpy.outer(self.pc, self.pc)
            + self.ccovmu * numpy.dot((self.weights * artmp.T), artmp) / self.sigma**2
        )

        self.sigma *= numpy.exp(
            (numpy.linalg.norm(self.ps) / self.chiN - 1.0) * self.cs / self.damps
        )

        self.diagD, self.B = numpy.linalg.eigh(self.C)
        indx = numpy.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD

    def computeParams(self, params):
        r"""Computes the parameters depending on :math:`\lambda`. It needs to
        be called again if :math:`\lambda` changes during evolution.
        :param params: A dictionary of the manually set parameters.
        """
        self.mu = params.get("mu", int(self.lambda_ / 2))
        rweights = params.get("weights", "superlinear")
        if rweights == "superlinear":
            self.weights = log(self.mu + 0.5) - numpy.log(numpy.arange(1, self.mu + 1))
        elif rweights == "linear":
            self.weights = self.mu + 0.5 - numpy.arange(1, self.mu + 1)
        elif rweights == "equal":
            self.weights = numpy.ones(self.mu)
        else:
            raise RuntimeError("Unknown weights : %s" % rweights)

        self.weights /= sum(self.weights)
        self.mueff = 1.0 / sum(self.weights**2)

        self.cc = params.get("ccum", 4.0 / (self.dim + 4.0))
        self.cs = params.get("cs", (self.mueff + 2.0) / (self.dim + self.mueff + 3.0))
        self.ccov1 = params.get("ccov1", 2.0 / ((self.dim + 1.3) ** 2 + self.mueff))
        self.ccovmu = params.get(
            "ccovmu",
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.dim + 2.0) ** 2 + self.mueff),
        )
        self.ccovmu = min(1 - self.ccov1, self.ccovmu)
        self.damps = (
            1.0 + 2.0 * max(0.0, sqrt((self.mueff - 1.0) / (self.dim + 1.0)) - 1.0) + self.cs
        )
        self.damps = params.get("damps", self.damps)
