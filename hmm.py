__author__ = 'RodXander'

from sys import float_info
from random import uniform
import os
import math
import cPickle

import scipy.cluster.vq as vq
import scipy.ndimage.measurements as msr
import PyQt4.QtCore as QtCore
import scipy.io.wavfile as wav

import mixtures as mx
from logic.mfcc import mfcc
from logic.utils import *


percentage_of_exponent = 0.9
under_threshold = 10 ** (percentage_of_exponent * float_info.min_10_exp)
upper_threshold = 10 ** (percentage_of_exponent * float_info.max_10_exp)
one_member_cluster_variance = 0.1E-3


class RecognizeModel(QtCore.QObject):
    progress_signal = QtCore.pyqtSignal(int, int, int)
    end_proccess = QtCore.pyqtSignal()

    def __init__(self, hmms, audios):
        QtCore.QObject.__init__(self)
        self.hmms = hmms
        self.audios = audios
        self.cancel = False

    def recognize(self):
        hmms = []
        for model_address in self.hmms:
            hmms.append(cPickle.load(file(model_address)))

        audios = []
        for audio_address in self.audios:
            (rate, sig) = wav.read(audio_address)
            # Some files are so perfect in the absent of noise that some parts of signal are zero.
            # This is not a good thing, because the frames with all zero will return an fft with all zero,
            # which when we calculate the log will be -inf. Extremely bad news!!!!!!!!
            nonzeros_in_signal = np.where(sig != 0)[0]
            audios.append(remove_nan_rows(mfcc(sig[nonzeros_in_signal[0]:(nonzeros_in_signal[-1] + 1)], samplerate=rate, winstep=0.015)))

        increase_for_recognizing = 100. / len(self.audios)
        current_progress_bar_value = 0

        results = []
        for index_audio, seq in enumerate(audios):
            if self.cancel:
                self.progress_signal.emit(0, -1, -1)
                exit()

            this_audio_results = []
            for model in hmms:
                this_audio_results.append(model.prob_emit_sequence(seq))

            best_model_index = this_audio_results.index(max(this_audio_results))
            results.append(best_model_index)

            current_progress_bar_value += increase_for_recognizing
            self.progress_signal.emit(current_progress_bar_value, index_audio, best_model_index)

        self.progress_signal.emit(100, -1, -1)
        self.end_proccess.emit()

        return results


class TrainModel(QtCore.QObject):
    # Although this is an intrusion in the bussiness logic from the UI. I need it to obtain a real feedback on how is
    # advancing the process.
    progress_signal = QtCore.pyqtSignal(int)
    end_proccess = QtCore.pyqtSignal()

    def __init__(self, model, audios, location, nsegmentations=150, niters_bw=10,
                 score_method='forward', apply_baum_welch=True):
        """Creates the object that allows the training of a model through a given training set

            :param model: Required model to train its parameters
            :param audios: Required sequences of observations for training
            :param apply_baum_welch: Optional value that determines if should be applied the baum-welch reestimation procedure.
                If true the training sequence will only be segmented, first uniformly and later using the Viterbi algorithm a
                number of times. If false then a baum-welch reestimation will be applied.
            :param score_method: Optional value that specifies the kind of scoring that will be used later for the recognition process.
            :param nsegmentations: Optional number of the maximum number of segmentations performed from the a given model.
            :param niters_bw: Optional number of iterations for the baum-welch procedure.
            """
        QtCore.QObject.__init__(self)
        self.best_model = model
        self.best_scores = 0
        self.best_scores_sum = 0
        self.audios = audios
        self.apply_baum_welch = apply_baum_welch
        self.score_method = score_method
        self.nsegmentations = nsegmentations
        self.niters_bw = niters_bw

        self.location = location
        self.current_iteration = 0

        self.training_seqs = []

        self.jump_between_improvements = 0.05
        self.cancel = False

    def train(self):
        for audio_address in self.audios:
            (rate, sig) = wav.read(audio_address)
            # Some files are so perfect in the absent of noise that some parts of signal are zero.
            # This is not a good thing, because the frames with all zero will return an fft with all zero,
            # which when we calculate the log will be -inf. Extremely bad news!!!!!!!!
            nonzeros_in_signal = np.where(sig != 0)[0]
            self.training_seqs.append(remove_nan_rows(mfcc(sig[nonzeros_in_signal[0]:(nonzeros_in_signal[-1] + 1)], samplerate=rate, winstep=0.015)))

        if self.apply_baum_welch:
            increase_for_segmentations = float(100 - self.niters_bw) / self.nsegmentations
        else:
            increase_for_segmentations = float(100) / self.nsegmentations

        i = 0
        while i < self.nsegmentations:
            if i == 0:
                # Using an initial uniform segmentation due the lack of information about the states' position
                segmentation = state_segmentation(self.best_model, self.training_seqs, method='uniform')
            else:
                # We already have a model, so let's try a segmentation with it.
                segmentation = state_segmentation(self.best_model, self.training_seqs)

            # The method returns true if the best model changed otherwise I don't continue
            i = self.recompute_centroids(segmentation, increase_for_segmentations, i)

        best_seg_model = self.best_model

        if self.apply_baum_welch:
            # Start the reestimation procedure for the best segmentation found
            next_model = best_seg_model
            for j in range(self.niters_bw):
                new_model = baum_welch_reestimation(next_model, self.training_seqs)
                # I don't care if there was an improvement or not because we alway make the number iterations ordered

                self.compare_models(new_model)

                next_model = HMM(new_model.nstates, new_model.obsdim,
                                 nmixtcomponents=[new_model.states[state].ncomponents for state in
                                                  range(new_model.nstates)],
                                 smaller_variance_allowed=new_model.smaller_variance_allowed)

                next_model.estimate_parameters(state_segmentation(new_model, self.training_seqs),
                                               number_of_training_seqs=len(self.training_seqs),
                                               old_centroids=[
                                                   [component.mean for component in new_model.states[state].components]
                                                   for state in range(new_model.nstates)])

                self.current_iteration += 1
                self.progress_signal.emit(self.current_iteration)

        self.progress_signal.emit(100)
        self.end_proccess.emit()

        cPickle.dump(self.best_model, file(self.location, 'w'))
        cPickle.dump(best_seg_model, file(
            os.path.dirname(self.location) + os.sep + '__' + os.path.basename(self.location), 'w'))

    def recompute_centroids(self, segmentation, increase_for_segmentations, i):
        end_random_calculation = 0
        while i < self.nsegmentations and end_random_calculation < uniform(0, 1):
            if self.cancel:
                self.progress_signal.emit(0)
                exit()

            next_model = HMM(self.best_model.nstates, self.best_model.obsdim, self.best_model.nmixtcomponents,
                             smaller_variance_allowed=self.best_model.smaller_variance_allowed)
            # Estimating the parameters again using the viterbi algorithm (instead of an uniform one) to obtain the
            # inner sequence of states for each sequence of training to segment them into states.
            next_model.estimate_parameters(segmentation, number_of_training_seqs=len(self.training_seqs))

            improvement = self.compare_models(next_model)
            # If an improvement has occured I increse the probability of ending this random search of better means.
            # In case of exiting the I'll use the new best to segment and get in here again.
            # The main point is that for every segmentation I don't settle for the first improvement, instead I
            # keep looking for a better one. Until the number of improvements is so elevated that the variable
            # end_random_calculation will be to high and the loop will exit.
            if improvement:
                end_random_calculation += self.jump_between_improvements

            i += 1

            self.current_iteration += increase_for_segmentations
            self.progress_signal.emit(self.current_iteration)

        # Smoothing the result to get be best possible
        i = self.smooth_centroids(segmentation, increase_for_segmentations, i)

        return i

    def smooth_centroids(self, segmentation, increase_for_segmentations, i):
        improvement = True
        while i < self.nsegmentations and improvement:

            next_model = HMM(self.best_model.nstates, self.best_model.obsdim,
                             nmixtcomponents=[self.best_model.states[state].ncomponents for state in
                                              range(self.best_model.nstates)],
                             smaller_variance_allowed=self.best_model.smaller_variance_allowed)

            next_model.estimate_parameters(segmentation, number_of_training_seqs=len(self.training_seqs),
                                           old_centroids=[[component.mean for component in
                                                           self.best_model.states[state].components] for state in
                                                          range(self.best_model.nstates)])

            improvement = self.compare_models(next_model)
            i += 1

            self.current_iteration += increase_for_segmentations
            self.progress_signal.emit(self.current_iteration)

        return i

    def compare_models(self, next_model):
        improvement = False

        scores = np.array(
            [next_model.prob_emit_sequence(seq, score_method=self.score_method) for seq in self.training_seqs])
        scores_sum = sum(scores)

        nwins = np.count_nonzero(scores > self.best_scores)
        nloss = np.count_nonzero(scores < self.best_scores)

        # If this model beats to our best model to the date, then it replaces it.
        # Beat means that it has an overall larger probability

        if self.best_scores_sum < scores_sum or (self.best_scores_sum == scores_sum and nwins > nloss):
            improvement = True
            self.best_model = next_model
            self.best_scores = scores
            self.best_scores_sum = scores_sum

        return improvement


def baum_welch_reestimation(curr_model, training_seqs):
    """Applies the Baum-Welch reestimations formulas for the given model using the sequences of observations

    :param curr_model: Model to reestimate its parameters
    :param training_seqs: Multiple (or single) sequences of observations to train the model with
    :param curr_model: Optional value that if different from None is an int that specifies the max length of the jump
        from a current state into another in the future.
    :return: Nothing, but the model is updated
    """
    new_model = HMM(curr_model.nstates, curr_model.obsdim,
                    nmixtcomponents=[curr_model.states[i].ncomponents for i in range(curr_model.nstates)],
                    smaller_variance_allowed=curr_model.smaller_variance_allowed)

    decimal_transition_matrix = np.zeros((curr_model.nstates, curr_model.nstates), dtype=d)

    for sequence in training_seqs:
        length_seq = len(sequence)

        forward_vars, scale_factors = curr_model.forward_algorithm(sequence)
        backward_vars = curr_model.backward_algorithm(sequence, scale_factors)

        # Used to cancel the effect of the scaling procedure
        scale_factors_prod = prod(scale_factors, dtype=d)

        for i in range(curr_model.nstates):
            # Computing the numerator for the reestimation formula of the transition probability
            decimal_transition_matrix[i, i] += scale_factors_prod * \
                                               sum([d(forward_vars[i, time]) * d(curr_model.transition_matrix[i, i]) *
                                                    d(curr_model.states[i].get_density(sequence[time + 1])) *
                                                    d(backward_vars[i, time + 1]) for time in range(length_seq - 1)])

            if i < curr_model.nstates - 1:
                decimal_transition_matrix[i, i + 1] += scale_factors_prod * \
                                                       sum([d(forward_vars[i, time]) *
                                                            d(curr_model.transition_matrix[i, i + 1]) *
                                                            d(curr_model.states[i + 1].get_density(sequence[time + 1]))
                                                            * d(backward_vars[i + 1, time + 1])
                                                            for time in range(length_seq - 1)])

            for component in range(curr_model.states[i].ncomponents):
                # This list will end up containing the probabilities of being in state 'i' accounted for the component
                # 'm' of its mixture at a given time 't'
                gamma_i_comp = []
                for time in range(length_seq):

                    gamma_i_comp.append(forward_vars[i, time] * backward_vars[i, time] / sum(
                        [forward_vars[state, time] * backward_vars[state, time] for state in
                         range(curr_model.nstates)]) * \
                                        (curr_model.states[i].get_density(sequence[time], component=component) /
                                         curr_model.states[i].get_density(sequence[time])))

                    gamma_i_comp[-1] = _fix_threshold(gamma_i_comp[-1], under_thd=under_threshold, upper_thd=upper_threshold, nan=under_threshold)

                    # Computing the numerators of the reestimation formulas for the mixture's parameters
                    new_model.states[i].coefficients[component] += gamma_i_comp[time]
                    new_model.states[i].components[component].mean += gamma_i_comp[time] * sequence[time]
                    new_model.states[i].components[component].covariance_matrix += gamma_i_comp[time] * np.dot(
                        (sequence[time] - curr_model.states[i].components[component].mean).T,
                        sequence[time] - curr_model.states[i].components[component].mean)

    # Adding the denominators to reestimation formulas
    for state in range(new_model.nstates):
        prob_states_sum = sum(decimal_transition_matrix[state])
        # Sometimes, mostly when we have few information and many states the reestimation formula gives zero to all
        # values in the same row. We assume here that in that case this state is a transient one and we must get out of
        # it as soon as possible.
        if prob_states_sum == 0:
            if state != new_model.nstates - 1:
                decimal_transition_matrix[state, state + 1] = d(1)
            else:
                decimal_transition_matrix[state, state] = d(1)
        else:
            # Computing and adding the denominator to the transition reestimation formula to obtain the reestimated values
            decimal_transition_matrix[state] /= prob_states_sum

        for component in range(new_model.states[state].ncomponents):
            # Doing the same for the reestimation formulas for the mixture's parameters. Notice how the denominators
            # for the mean and covariance matrix is the numerator of the coefficient's reestimation formula.
            new_model.states[state].components[component].mean /= new_model.states[state].coefficients[component]
            new_model.states[state].components[component].covariance_matrix /= new_model.states[state].coefficients[
                component]

        new_model.states[state].coefficients /= sum(new_model.states[state].coefficients)


    # Saving the computed values for the transition matrix
    new_model.transition_matrix = decimal_transition_matrix.astype(float)
    # Fixing the probabilities that are to small or zero, becuase this will prevent the pass through the states.
    for i in range(new_model.nstates):
        new_model.transition_matrix[i, i] = _fix_threshold(new_model.transition_matrix[i, i], under_thd=under_threshold, nan=under_threshold)

        if i < new_model.nstates - 1:
            new_model.transition_matrix[i, i + 1] = _fix_threshold(new_model.transition_matrix[i, i + 1], under_thd=under_threshold, nan=under_threshold)

    new_model.end_building()
    _fix_singular_matrixes(new_model, curr_model)

    return new_model


def _fix_singular_matrixes(curr_model, old_model):
    """Change the singular covariances matrixes of the current model with those of the old one suppose not to be singular

    The need of making these operation relies in the fact the matrixes with determinant equal to zero are singular, and
    at the same time these matrixes cannot be used in the multivariate normal distributions used in the components of
    the state's mixtures. Preventing the model from calculate a density value for a given observation and
    making it useless.

    :param curr_model: New model which covariances matrixes are unknown to be or not singular
    :param old_model: Old model which covariances matrixes are not singular
    :return: Nothing, but the model is changed.
    """
    for state in range(curr_model.nstates):
        for component in range(curr_model.states[state].ncomponents):
            # A determinant less than zero will not allow to calculate the density of this mixture because the formulae
            # needs the squeare root of the determinant. If this is equal to zero then there not exist an inverse
            # matrix which we need also in the formulae. So both scenarios have to be taken care away.
            if curr_model.states[state].components[component].det_covariance_matrix <= 0:
                new_cov_matrix = np.zeros((curr_model.obsdim, curr_model.obsdim))
                curr_cov_matrix = curr_model.states[state].components[component].covariance_matrix

                for feature in range(curr_model.obsdim):
                    if curr_cov_matrix[feature, feature] < curr_model.smaller_variance_allowed:
                        new_cov_matrix[feature, feature] = curr_model.smaller_variance_allowed
                    else:
                        new_cov_matrix[feature, feature] = curr_cov_matrix[feature, feature]

                curr_model.states[state].components[component].covariance_matrix = new_cov_matrix
                # Recalculate these values for the new covariance marix with determinant positive.
                curr_model.states[state].components[component].end_building()


def _fix_threshold(value, under_thd=None, upper_thd=None, nan=None):
    if under_thd and value < under_thd:
        return under_thd
    if upper_thd and value > upper_thd:
        return upper_thd
    if nan and math.isnan(value):
        return nan

    return value

def state_segmentation(hmm, training_seqs, method='viterbi', multiple=True):
    """Segments multiples (or a single) sequences of observations into a given groups of states according to the method specified

    :param hmm: Required model from which we'll extract the required information to make the segmentation
    :param training_seqs: Required multiple sequences of observations.
    :param method: Optional parameter that specified the method to segment. The method 'viterbi' uses this algorithm to
        find the best path of state for the sequence given the HMM. Also supports 'uniform' that separates the sequence
        in even parts each belonging to different states.
    :param multiple: Optional value that flags if the sequence is compound by multiple sequences of observations
        instead of a single one.
    :return: A sequence of the sequence of observations that belong to any given state
    """
    if multiple:
        result = None
        for sequence in training_seqs:
            segm_states = single_state_segmentation(hmm, sequence, method)
            if result is None:
                result = segm_states
            else:
                for i in range(hmm.nstates):
                    result[i] = np.concatenate((result[i], segm_states[i]))

        return result

    else:
        return single_state_segmentation(hmm, training_seqs, method)


def single_state_segmentation(hmm, training_seq, method='viterbi'):
    """Segments a sequence of observation into a given groups of states according to the method specified

    :param hmm: Required model from which we'll extract the required information to make the segmentation
    :param training_seq: Required sequence of observations.
    :param method: Optional parameter that specified the method to segment. The method 'viterbi' uses this algorithm to
    find the best path of state for the sequence given the HMM. Also supports 'uniform' that separates the sequence
    in even parts each belonging to different states.
    :return: A sequence of the sequence of observations that belong to any given state
    """
    sequence = np.array(training_seq)
    length_seq = len(sequence)

    if hmm.nstates == 1:  # If only one state, don't performs all the calculations
        segm_states = [sequence]

    elif str.lower(method) == 'uniform':
        sizes = np.array([length_seq // hmm.nstates] * hmm.nstates)
        sizes[r.sample(range(hmm.nstates), length_seq % hmm.nstates)] += 1

        segm_states = [sequence[sum(sizes[: i]):sum(sizes[: i + 1])] for i in range(hmm.nstates)]

    elif str.lower(method) == 'viterbi':
        prob, opt = hmm.viterbi_algorithm(sequence, compute_path=True)
        segm_states = [sequence[opt == i] for i in range(hmm.nstates)]

    else:
        raise ValueError('Method parameter must contain only these two options: \'viterbi\' or \'uniform\'. See help')

    return segm_states


class HMM:
    def __init__(self, nstates, obsdim, nmixtcomponents, smaller_variance_allowed=None):
        """Creates an object that represents a Hidden Markov Model (HMM)

        :param nstates: Number of states for the HMM
        :param obsdim: Dimension of the observations used with this HMM
        :param nmixtcomponents: Optional int or list. If the parameter is a number it's used to denote how many components
            have every mixture. If it is a list this must be of the same length as the number of states and should
            contain the number of components to use for the mixture of the respective state. By default is the integer 5.
            This specially useful when the model created it's going to be used in a reestimation procedure. Otherwise
            the model is created with default values.
        :return Return the HMM object
        """
        self.nstates = nstates
        self.obsdim = obsdim
        self.nmixtcomponents = nmixtcomponents

        if smaller_variance_allowed is None:
            self.smaller_variance_allowed = nroot(under_threshold, obsdim)
        else:
            self.smaller_variance_allowed = smaller_variance_allowed

        # Notice I'm creating a Bakis-model HMM. Which mean that the process can only go forward in the states
        # (i.e. the process begin in a default state always and keep advancing until it reach a final state)
        # Also notice that this particular Bakis-model just allow jumps from continuous states or itself
        self.transition_matrix = np.zeros((self.nstates, self.nstates))

        if type(nmixtcomponents) is int:
            self.states = np.array([mx.Mixture(nmixtcomponents, obsdim) for i in range(nstates)])
        elif type(nmixtcomponents) is list and len(nmixtcomponents) == nstates:
            self.states = np.array([mx.Mixture(nmixtcomponents[i], obsdim) for i in range(nstates)])
        else:
            raise ValueError(
                "The parameter 'nmixtcomponents' must be an int or a tuple of length equal to the parameter 'nstates'. See help")

    def forward_algorithm(self, training_seq):
        """Computes the values of the forward variables for the HMM training

        :param training_seq: Sequence of observations from which we extract the values of the forward variables
        :return: The matrix of forward variables scaled (state x time) and the sequence of scales factor for any time
        """
        length_seq = len(training_seq)

        scale_factors = [0.] * length_seq
        scale_factors[0] = self.states[0].get_density(training_seq[0])

        forward_vars = np.ones((self.nstates, length_seq)) * under_threshold
        forward_vars[0, 0] = self.states[0].get_density(training_seq[0]) / scale_factors[0]

        for curr_time in range(1, len(training_seq)):
            for curr_state in range(self.nstates):

                # Calculate alpha(curr_state, curr_time) (i.e. forward variable in time 'curr_time' for state 'curr_state')
                forward_vars[curr_state, curr_time] = sum(
                    [forward_vars[i, curr_time - 1] * self.transition_matrix[i, curr_state] for i in
                     range(self.nstates)]) * self.states[curr_state].get_density(training_seq[curr_time])

                forward_vars[curr_state, curr_time] = _fix_threshold(forward_vars[curr_state, curr_time], under_thd=under_threshold, upper_thd=upper_threshold)

            scale_factors[curr_time] = sum(forward_vars[:, curr_time])  # Saving the 'curr_time' scale factor
            forward_vars[:, curr_time] /= scale_factors[curr_time]  # Applying the scale factor

        return forward_vars, scale_factors

    def backward_algorithm(self, training_seq, scale_factors):
        """Computes the values of the backward variables for the HMM training

        :param training_seq: Required sequence of observations from which we extract the values of the backward variables
        :param scale_factors: Required sequence of scaling factors to use for the backward variables
        :return: The matrix of backward variables scaled (state x time)
        """
        length_seq = len(training_seq)

        backward_vars = np.zeros((self.nstates, length_seq))
        backward_vars[:, -1] = 1. / scale_factors[-1]

        for curr_time in range(length_seq - 2, -1, -1):
            for curr_state in range(self.nstates):
                # Calculate betha(curr_state, curr_time) (i.e. backward variable in time 'curr_time' for state 'curr_state')
                backward_vars[curr_state, curr_time] = sum([self.transition_matrix[curr_state, i] *
                                                            self.states[i].get_density(training_seq[curr_time + 1]) *
                                                            backward_vars[i, curr_time + 1] for i in
                                                            range(self.nstates)])

                backward_vars[curr_state, curr_time] = _fix_threshold(backward_vars[curr_state, curr_time], under_thd=under_threshold, upper_thd=upper_threshold)

            backward_vars[:, curr_time] /= scale_factors[curr_time]
            backward_vars[:, curr_time][backward_vars[:, curr_time] == float('inf')] = upper_threshold

        return backward_vars

    def viterbi_algorithm(self, sequence, compute_path=False):
        """Computes the probability of a given sequence of being generated by the HMM

        :param sequence: Sequence to be considered as generated
        :param compute_path: Optional value which denotes if the viterbi algorithm should compute the best path
        :return: A pair: The probability of the sequence of being generated by the HMM and the sequence
        of states who'd achieve this probability. Only if compute_path is True, otherwise is only the probability.
        """
        length_seq = len(sequence)

        lasts_vit_vars = [float('-inf')] * self.nstates
        lasts_vit_vars[0] = log10(self.states[0].get_density(sequence[0]))

        if compute_path:
            # Contains at the cell (i,j) the state that precedes state i in the optimal path of length j that ends at i
            optimal_vars = np.zeros((self.nstates, len(sequence)), dtype=int)

        for index_obs in range(1, length_seq):

            new_vit_vars = [0.] * self.nstates
            for i in range(self.nstates):
                # Store all possible transitions to curr_state from the previous ones to later choose the best one
                transitions = [lasts_vit_vars[j] + self.log_transition_matrix[j][i] for j in range(self.nstates)]

                new_vit_vars[i] = max(transitions) + log10(self.states[i].get_density(sequence[index_obs]))

                if compute_path:
                    optimal_vars[i, index_obs] = transitions.index(max(transitions))

            lasts_vit_vars = new_vit_vars

        # Obtaining the optimal path
        if compute_path:
            optimal_path = np.zeros(length_seq)
            optimal_path[-1] = lasts_vit_vars.index(max(lasts_vit_vars))

            for i in range(length_seq - 1, 0, -1):
                optimal_path[i - 1] = optimal_vars[optimal_path[i], i]

            return max(lasts_vit_vars), optimal_path
        else:
            return max(lasts_vit_vars)

    def estimate_parameters(self, segmented_obs, number_of_training_seqs=1, old_centroids=False):
        """Estimates the parameters of the hmm model from the segmented sequence of observations

        Using the clustering method: k-means. This procedure obtains as many groups of values from 'training_seq' as
        specified by 'ncomponents' and from them extract the information necessary to initialize the values of the mixture
        properly. The coefficients of each mixture component are obtain by the percentage it represents from the total.
        The mean vector is obtain from the inner functioning of the k-means procedure and the covariance matrix is
        calculated, but only its diagonal (which means that the covariances amongst the individual random variables are not
        taken into consideration, only the variances).
        Last the transition matrix is also estimated in a similar manner of that of the coefficients

        :param segmented_obs: Required array containing the sequences of observations that belong to each state.
            The index of each sequence has to be the same of the state index.
        :param number_of_training_seqs: Optional value that represents the number of training sequences used for the
            segmentation problem.
        :param old_centroids: Optional value that determines if new means are computed for the components of the mixture
            or the same the model provides are used to initially approach the segmented data.
        :return: Nothing, but the hmm will be updated
        """
        for i in range(self.nstates):
            if len(segmented_obs[i]) == 0:
                # Some times the clustering of the states leaves one of them empty so it would be impossible to
                # estimate anything. So we initialize its parameters in a way it be as improbable as possible
                # Notice this is different from empty components, which we can just delete
                self.states[i].return_minimum_density = True

                self.transition_matrix[i, i] = under_threshold
                if i < self.nstates - 1:
                    self.transition_matrix[i, i + 1] = 1.

                continue

            # Saving the standard deviations of each feature to recover later the true means
            standard_deviations = [msr.standard_deviation(segmented_obs[i][:, j]) for j in range(self.obsdim)]

            # Sometimes the kmeans2 throw an exception related to the impossibility of making the
            # Cholesky descomposition due to a very small number of observations for state. So we use a simpler method
            # of initialization for the kmeans
            try:
                if not old_centroids:
                    # Getting the features of the observations to unitary variance. It improves the kmeans
                    means, labels = vq.kmeans2(vq.whiten(segmented_obs[i]), self.states[i].ncomponents,
                                               minit='random')
                else:
                    means, labels = vq.kmeans2(vq.whiten(segmented_obs[i]), np.array([
                        [float(feature) / standard_deviations[index] for index, feature in
                         enumerate(old_centroids[i][j][0])] for j in
                        range(self.states[i].ncomponents)]), minit='matrix')
            except np.linalg.linalg.LinAlgError:
                # Instead of giving the initial centroids or trying to approximate them, I just take values from the data
                means, labels = vq.kmeans2(vq.whiten(segmented_obs[i]), self.states[i].ncomponents,
                                           minit='points')

            # Fixing the means to get the ones according to the real values
            means[:] *= standard_deviations

            # Removing the centroids whose cluster has no elements
            means, labels = delete_clusters_less_than(segmented_obs[i], means, labels)

            # Update the number of components for this state after making the kmeans and determining the clusters
            self.states[i].ncomponents = len(means)

            self.states[i].components = []
            self.states[i].coefficients = []

            for component in range(self.states[i].ncomponents):
                # Count how many observations belong to a cluster 'i' and established its coefficient as the ratio
                # between this number and the number of observations
                self.states[i].coefficients.append((labels == component).sum() / float(len(labels)))

                # Obtain the variances associated with each dimension of the vectors associated with a given cluster
                # which give me a diagonal matrix with the variances for every cluster.
                # Notice like we don't accept values of variances equal to zero (which may happen if a dimension of the
                # vectors who belong to a state all share the same value) because it will make a singular matrix and
                # later we'll need to find its inverse. This is why when we are in the presence of a cluster of just
                # one element we make the variances equal to 0.0001 or more specifically the value of
                # one_member_cluster_variance.
                if np.count_nonzero(labels == component) > 1:
                    cov_matrixes = np.diag([nd.variance(segmented_obs[i][labels == component][:, j]) for j in
                                            range(self.obsdim)])
                else:
                    cov_matrixes = np.diag([one_member_cluster_variance] * self.obsdim)

                # Create the 'ncomponents' of the mixture.
                # This could be any elliptically symmetric density. Now we use multivariate normal distributions.
                self.states[i].components.append(mx.MultiNormalDist(means[component], cov_matrixes))

            # Count how many transitions make a given state and using this information we could estimate the ratio
            # of jumps to itself or another state. Notice that the last state remains with same transition probabilities
            # cause it can only jump to itself.
            if i != self.nstates - 1:
                transitions_in_i = len(segmented_obs[i])
                transitions_out_i = number_of_training_seqs
                total_transitions = float(transitions_in_i + transitions_out_i)

                self.transition_matrix[i, i] = transitions_in_i / total_transitions
                self.transition_matrix[i, i + 1] = transitions_out_i / total_transitions
            else:
                self.transition_matrix[i, i] = 1

            self.states[i].return_minimum_density = False

        self.end_building()

    def prob_emit_sequence(self, sequence, score_method='forward'):
        """Computes the probability that this model has emitted the specified sequence

        :param sequence: Required unknown sequence to determine its probability emission
        :param score_method: Optional method to use. The default value is 'Viterbi' which is fastest and probably better in
        most cases specifically if we're interested in the probability of the best state sequence. The 'forward' method
        uses this probabilities to determine this value. Have in mind that this method is slower and takes into account
        all possible state's sequences.
        :return: The probability of emission for the given sequence
        """
        if str.lower(score_method) == 'viterbi':
            best_prob = self.viterbi_algorithm(sequence)
            return best_prob
        elif str.lower(score_method) == 'forward':
            forward_vars, scale_factors = self.forward_algorithm(sequence)
            return d(sum(forward_vars[:, -1])) * prod(scale_factors, dtype=d)
        else:
            raise ValueError(
                'Method parameter must contain only these two options: \'viterbi\' or \'forward\'. See help')

    def end_building(self):
        """Call this method when you finished modifying the model. For instance during a baum-welch reestimation

        :return: Nothing, but the model will be finished. Which means mainly that some values will be computed and saved.
        """
        for mixture in self.states:
            self.log_transition_matrix = [[log10(self.transition_matrix[i, j]) for j in range(self.nstates)] for i in
                                          range(self.nstates)]
            mixture.end_building()