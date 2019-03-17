__author__ = 'RodXander'

import math
from sys import float_info
import hmm

import numpy as np
import scipy.linalg as linalg


percentage_of_exponent = 0.9
under_threshold = 10 ** (percentage_of_exponent * float_info.min_10_exp - 1)
upper_threshold = 1.


class Mixture:
    def __init__(self, ncomponents, ndimensions):
        """Creates an object that represents a linear combination of densities' distributions

        :param ncomponents: Number of mixtures' components
        :param ndimensions: Number of dimensions for the mixtures' components
        :return: An object that represents the specified mixture
        """
        # This attribute is used in case we want the mixtures to return a fixed small value.
        # This is useful during estimation where empty states may happen.
        self.return_minimum_density = False

        self.ncomponents = ncomponents
        self.ndimensions = ndimensions

        self.coefficients = [0.] * ncomponents
        # Create the 'ncomponents' of the mixture
        # This could be any elliptically symmetric density. But in this case we use multivariate normal distributions
        self.components = [MultiNormalDist([0.] * ndimensions, np.zeros((ndimensions, ndimensions))) for i in
                           range(ncomponents)]

    def end_building(self):
        for component in self.components:
            component.end_building()

    def get_density(self, event, component=None):
        """Compute the density of the passed event according to the mixture

        Sums over the densities of all the components multiply by the respective coefficient
        :param event: A numpy array that represents the vector whose density we'll compute
        :param component: Specified the mixture's component whose density we'll interested in. Its coefficient value will be applied
        :return: The "event" vector density
        """
        # Sometimes when there are no observations to estimate an state parameters. But still these will be consulted
        # about densities that's why we stablished this defaul and extremely low density for any observation.
        if self.return_minimum_density:
            return under_threshold

        if component is None:
            res = np.sum(
                [self.coefficients[i] * self.components[i].get_density(event) for i in range(self.ncomponents)])
        elif component >= self.ncomponents or component < 0:
            raise ValueError('Invalid component value.')
        else:
            res = self.coefficients[component] * self.components[component].get_density(event)

        # Making sure the density doesn't underpass the threshold stablished
        return hmm._fix_threshold(res, under_thd=under_threshold, upper_thd=upper_threshold)


class MultiNormalDist:
    def __init__(self, mean, covariance_matrix):
        """Creates an object that represents a multivariate normal distribution."

        :param mean: A numpy array that represents the mean vector of the distribution
        :param covariance_matrix: A numpy array that represents the covariance matrix of the distribution.
        Warning: Only contemplates diagonal covariances' matrixes!
        :return: An object that represents the specified multivariate normal distribution
        """
        self.ndimensions = len(mean)
        # Convert the input values to matrixes for later multiplications
        self.mean = np.array([mean])
        self.covariance_matrix = covariance_matrix

    def end_building(self):
        self.det_covariance_matrix = linalg.det(self.covariance_matrix)
        # Only exist inverse matrix if determinant is different from zero and the square root is only defined if
        # it's non negative.
        if self.det_covariance_matrix > 0:
            self.inv_covariance_matrix = linalg.inv(self.covariance_matrix)
            self.first_part_of_formulae = 1. / (
                6.283185307179586476925286766559 ** (self.ndimensions / 2.) * math.sqrt(self.det_covariance_matrix))

    def get_density(self, event):
        """Compute the density of the passed event according to the instance of MultivariateNormal

        :param event: A numpy array that represents the vector whose density we'll compute
        :return: The "event" vector density
        """
        if self.ndimensions != len(event):
            raise ValueError('Vector\'s dimension different from the distribution\'s dimension')

        # Convert the input values to matrixes for later mltiplications
        m_event = np.array([event])
        return self.first_part_of_formulae * (math.e ** (
            -0.5 * np.dot(np.dot(m_event - self.mean, self.inv_covariance_matrix), (m_event - self.mean).T)[0, 0]))