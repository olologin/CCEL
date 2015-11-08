"""CCEL estimator."""

from __future__ import division

import numbers
import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.base import ClassifierMixin
from sklearn.externals.six import with_metaclass
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, check_X_y, check_array, column_or_1d
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

from sklearn.ensemble.base import BaseEnsemble

from sklearn.externals import six

__all__ = ["CCELClassifier"]

MAX_INT = np.iinfo(np.int32).max

class Organizm:
    def __init__(self, n_samples, n_features, ps, pf,  random_state):
        self.random_state = random_state
        self.genome_features = np.zeros((n_features, ), dtype=np.bool8)
        self.genome_samples = np.zeros((n_samples, ), dtype=np.bool8)

        n_pick_samples = np.floor(n_samples*ps)
        n_pick_features = np.floor(n_features*pf)

        picked_samples = sample_without_replacement(n_samples,
                                                    n_pick_samples,
                                                    random_state=random_state)
        picked_features = sample_without_replacement(n_features,
                                                     n_pick_features,
                                                     random_state=random_state)
        self.genome_samples[picked_samples] = True
        self.genome_features[picked_features] = True

        self.cache_est_weight = 1
        self.cache_contribution = 0
        self.cache_predictions = None

    def mutation(self, pm):
        n_mutations_in_features = self.random_state.binomial(self.genome_features.shape[0], pm)
        n_mutations_in_samples = self.random_state.binomial(self.genome_samples.shape[0], pm)

        mutated_samples = sample_without_replacement(self.genome_samples.shape[0],
                                                     n_mutations_in_samples,
                                                     random_state=self.random_state)
        mutated_features = sample_without_replacement(self.genome_features.shape[0],
                                                     n_mutations_in_features,
                                                     random_state=self.random_state)
        self.genome_samples[mutated_samples] = ~self.genome_samples[mutated_samples]
        self.genome_features[mutated_features] = ~self.genome_features[mutated_features]

    def crossover(self, first, second):
        """ This object becomes a child for first and second object, just because i don't want
          to allocate new instance, reuse current"""

        self.genome_features = first.genome_features == second.genome_features
        self.genome_samples  = first.genome_samples  == second.genome_samples
        samples_difference_idx, = np.where(first.genome_samples ^ second.genome_samples)
        features_difference_idx, = np.where(first.genome_features ^ second.genome_features)
        if samples_difference_idx.shape[0] > 0:
            samples_inherited  = self.random_state.choice(samples_difference_idx,
                                                         np.floor(samples_difference_idx.shape[0]/2), replace=False)
            self.genome_samples[samples_inherited] = True
        if features_difference_idx.shape[0] > 0:
            features_inherited = self.random_state.choice(features_difference_idx,
                                                          np.floor(features_difference_idx.shape[0]/2), replace=False)
            self.genome_features[features_inherited] = True

class BaseCCEL(with_metaclass(ABCMeta, BaseEnsemble)):
    """Base class for CCEL meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 tmax, # maximum number of generations
                 po, # p of choosing object
                 pf, # p of choosing feature
                 pm, # p of mutation
                 N2, # how much left from prev generation
                 N1, # size of offsprings set
                 N0, # from intermidiate to next generation
                 d1, # number of generation on which we compute average contribution of classifier
                 eps1, # if contribution with and without classifier differs less than eps1 - remove classifier
                 d2, # check stagnation criterion back to d2 generations
                 eps2, #eps for stagnation criterion
                 d3, # for termination
                 eps3, # same
                 base_estimator=None,
                 min_estimators=1,
                 min_samples=0.25,
                 min_features=0.25,
                 bootstrap=True,
                 bootstrap_features=False,
                 compute_population_predictions = None,
                 get_estimator_fitness_func = None,
                 random_state=None,
                 verbose=0):
        super(BaseCCEL, self).__init__(
            base_estimator=base_estimator,
            n_estimators=min_estimators)

        self.min_estimators = min_estimators
        self.tmax = tmax
        self.ps = po
        self.pf = pf
        self.pm = pm
        self.min_samples = min_samples
        self.min_features = min_features
        assert N1 > N0, "size of offspring set  must be greater than population size"
        assert N2 <= N0, "N2 must be less than size of initial population"
        self.N2 = N2
        self.N1 = N1
        self.N0 = N0

        self.d1 = d1
        self.eps1 = eps1
        self.d2 = d2
        self.eps2 = eps2
        self.d3 = d3
        self.eps3 = eps3
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.verbose = verbose
        self.compute_population_predictions = compute_population_predictions
        self.get_estimator_fitness_func = get_estimator_fitness_func
        self.n_jobs = 2

    def fit(self, X, y, sample_weight=None):
        random_state = check_random_state(self.random_state)

        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])

        # Remap output
        n_samples, self.n_features_ = X.shape
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if isinstance(self.min_samples, (numbers.Integral, np.integer)):
            min_samples = self.min_samples
        else:  # float
            min_samples = int(self.min_samples * X.shape[0])

        if not (0 < min_samples <= X.shape[0]):
            raise ValueError("min_samples must be in (0, n_samples]")

        if isinstance(self.min_features, (numbers.Integral, np.integer)):
            min_features = self.min_features
        else:  # float
            min_features = int(self.min_features * self.n_features_)

        if not (0 < min_features <= self.n_features_):
            raise ValueError("min_features must be in (0, n_features]")

        if self.min_estimators <= 0:
            raise ValueError("min_estimators must be greater than 0")

        support_predict_proba = hasattr(self.base_estimator_, "predict_proba")
        support_sample_weight = has_fit_parameter(self.base_estimator_,
                                                      "sample_weight")
        if not support_sample_weight and sample_weight is not None:
            raise ValueError("The base estimator doesn't support sample weight")

        self.estimators_ = []

        accuracy_per_generation = np.zeros((self.tmax,), dtype=float)
        _best_accuracy = 0
        _best_organizms = []
        _populations = []
        _contributions = []
        _estimators = [self._make_estimator(append=False) for _ in range(self.n_jobs)]
        _offsprings = [Organizm(n_samples, self.n_features_, self.ps, self.pf, np.random.RandomState(random_state.randint(100))) for _ in range(self.N1)]

        def _append_population():
            population = [Organizm(n_samples, self.n_features_, self.ps, self.pf, np.random.RandomState(random_state.randint(100))) for _ in range(self.N0)]
            _populations.append(population)
            _contributions.append(np.arange(self.d1, dtype=float))
            self.compute_population_predictions(X, y, sample_weight, _estimators[0], population)

        def _remove_population(idx):
            del _populations[idx]
            del _contributions[idx]

        def genchoices():
            res = set()
            while len(res) < self.N1:
                first = random_state.randint(0, self.N0-1)
                second = random_state.randint(first+1, self.N0)
                res.add((first, second))
            return list(res)

        for _ in range(self.min_estimators):
            _append_population()

        step_N0 = int(self.N0 / self.n_jobs)
        step_N1 = int(self.N1 / self.n_jobs)
        with Parallel(n_jobs=self.n_jobs, backend="threading") as parallel:

            for generation in range(self.tmax):
                idx = 0
                print("Generation #{0}".format(generation))
                while idx < len(_populations):
                    population = _populations[idx]
                    fitness_func = self.get_estimator_fitness_func(
                        [o[0] for i, o in enumerate(_populations) if i != idx],
                        y,
                        sample_weight)
                    # crossover
                    parallel(delayed(_compute_cache_accuracy_contribution)(fitness_func, population[idx:idx+step_N0])
                             for idx in range(0, self.N0, step_N0))

                    choices = genchoices()
                    # compute_population_predictions, pm, fitness_func, X, y, sample_weight, estimator, choices, population, offsprings
                    parallel(delayed(_offspring_routines)(self.compute_population_predictions,
                                                          self.pm,
                                                          fitness_func,
                                                          X,
                                                          y,
                                                          sample_weight,
                                                          _estimators[int(idx/step_N1)],
                                                          choices[idx:idx+step_N1],
                                                          population,
                                                          _offsprings[idx:idx+step_N1])
                             for idx in range(0, self.N1, step_N1))

                    population.sort(reverse=True, key=lambda x:x.cache_accuracy)
                    a = sorted(population[:self.N2] + _offsprings, reverse=True, key=lambda x:x.cache_accuracy)
                    dead = population[self.N2:]
                    _populations[idx] = a[:self.N0]
                    _offsprings = dead+a[self.N0:]
                    accuracy_per_generation[generation] = max(accuracy_per_generation[generation],
                                                                 a[0].cache_accuracy)
                    print("Estimator #{0} from {1}, accuracy {2}, contribution {3}".format(idx, len(_populations), a[0].cache_accuracy, a[0].cache_contribution))
                    _contributions[idx][generation%self.d1] = a[0].cache_contribution

                    if len(_populations) > self.min_estimators and _contributions[idx].mean() < self.eps1:
                        print("Estimator #{0} removed, contribution was {1}".format(idx, _contributions[idx].mean()))
                        _remove_population(idx)
                    else:
                        idx += 1

                if (generation-self.d3-1 >=0 and
                            (accuracy_per_generation[generation-self.d2:generation].max() -
                                 accuracy_per_generation[:generation-self.d3].max()) < self.eps3):
                    print("Seems that adding new population doesn't helps, stopping...")
                    break

                if (generation-self.d2-1 >=0 and
                            (accuracy_per_generation[generation-self.d2:generation].max() -
                                 accuracy_per_generation[:generation-self.d2].max())  < self.eps2):
                    _append_population()
                    print("Stagnation, let's add new population")


        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_weights_ = []

        for population in _populations:
            organizm = population[0]
            estimator = self._make_estimator(append=False)
            estimator.random_state = organizm.random_state
            self.estimators_.append(estimator.fit(X[organizm.genome_samples,:][:,organizm.genome_features],
                                                  y[organizm.genome_samples],
                                                  sample_weight[organizm.genome_samples] if sample_weight is not None else None))
            self.estimators_features_.append(organizm.genome_features)
            self.estimators_weights_.append(organizm.cache_est_weight)

        return self

    def _validate_y(self, y):
        # Default implementation
        return column_or_1d(y, warn=True)

    # TODO Minimum features and minimum samples treating
    # TODO classifier weighting
    # Parallelization
    # Less parameters
    # max optimization

def _compute_cache_accuracy_contribution(fitness_func, population):
    accuracy_without_target = fitness_func()
    for organizm in population:
        organizm.cache_accuracy = fitness_func(organizm)
        organizm.cache_contribution = organizm.cache_accuracy-accuracy_without_target


def _offspring_routines(compute_population_predictions, pm, fitness_func, X, y, sample_weight, estimator, choices, population, offsprings):
    for offspring, (first, second) in zip(offsprings, choices):
        offspring.crossover(population[first], population[second])
        offspring.mutation(pm)
    compute_population_predictions(X, y, sample_weight, estimator, offsprings)
    _compute_cache_accuracy_contribution(fitness_func, offsprings)


def compute_cache_classifier_predictions(X, y, sample_weights, estimator, population):
    support_predict_proba = hasattr(estimator, "predict_proba")
    support_sample_weight = has_fit_parameter(estimator,
                                          "sample_weight")
    for organizm in population:
        estimator.random_state = organizm.random_state
        if sample_weights and support_sample_weight:
            estimator.fit(X[organizm.genome_samples,:][:,organizm.genome_features],
                           y[organizm.genome_samples], sample_weights[organizm.genome_samples])
        else:
            estimator.fit(X[organizm.genome_samples,:][:,organizm.genome_features],
                           y[organizm.genome_samples])

        if support_predict_proba:
            organizm.cache_predictions = estimator.predict_proba(X[:, organizm.genome_features])
        else:
            predictions = estimator.predict(X[:, organizm.genome_features])
            organizm.cache_predictions = np.zeros((predictions.shape[0], len(estimator.classes_)))
            for i in range(predictions.shape[0]):
                organizm.cache_predictions[i, predictions[i]] += 1


def get_classifier_fitness_func(ensemble_without_estimator, y_true, sample_weight=None):
    Y = np.sum(o.cache_est_weight * o.cache_predictions for o in ensemble_without_estimator)

    def fitness_func(organizm = None):
        if organizm == None:
            if np.shape(Y) == ():
                return 0

            ensemble_y = np.argmax(Y, axis=1)
            return accuracy_score(y_true, ensemble_y, sample_weight=sample_weight,
                                  normalize=True)

        Z = Y + (organizm.cache_est_weight * organizm.cache_predictions)
        ensemble_y = np.argmax(Z, axis=1)
        return accuracy_score(y_true, ensemble_y, sample_weight=sample_weight,
                              normalize=True)
    return fitness_func



class CCELClassifier(BaseCCEL, ClassifierMixin):
    def __init__(self,
                 tmax, # maximum number of generations
                 po, # p of choosing sample
                 pf, # p of choosing feature
                 pm, # p of mutation
                 N2, # how much left from prev generation
                 N1, # size of offsprings set
                 N0, # from intermidiate to next generation
                 d1, # number of generation on which we compute average contribution of classifier
                 eps1, # if contribution with and without classifier differs less than eps1 - remove classifier
                 d2, # check stagnation criterion back to d2 generations
                 eps2, #eps for stagnation criterion
                 d3, # for termination
                 eps3, # same
                 base_estimator=None,
                 min_estimators=1,
                 min_samples=0.25,
                 min_features=0.25,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None,
                 verbose=0):

        super(CCELClassifier, self).__init__(
                tmax, # maximum number of generations
                po, # p of choosing object
                pf, # p of choosing feature
                pm, # p of mutation
                N2, # how much left from prev generation
                N1, # size of offsprings set
                N0, # from intermidiate to next generation
                d1, # number of generation on which we compute average contribution of classifier
                eps1, # if contribution with and without classifier differs less than eps1 - remove classifier
                d2, # check stagnation criterion back to d2 generations
                eps2, #eps for stagnation criterion
                d3, # for termination
                eps3, # same
                base_estimator,
                min_estimators,
                min_samples,
                min_features,
                bootstrap,
                bootstrap_features,
                compute_cache_classifier_predictions,
                get_classifier_fitness_func,
                random_state,
                verbose)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(CCELClassifier, self)._validate_estimator(
            default=DecisionTreeClassifier())

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        return y

    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        check_is_fitted(self, "classes_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        n_samples, n_features = X.shape

        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        all_proba = np.zeros((X.shape[0], self.n_classes_))
        for estimator, features, weight in zip(self.estimators_, self.estimators_features_, self.estimators_weights_):
            if hasattr(estimator, "predict_proba"):
                proba_estimator = estimator.predict_proba(X[:, features])

                if self.n_classes_ == len(estimator.classes_):
                    all_proba += weight * proba_estimator

                else:
                    all_proba[:, estimator.classes_] += \
                        weight * proba_estimator[:, range(len(estimator.classes_))]

            else:
                # Resort to voting
                predictions = estimator.predict(X[:, features])

                for i in range(n_samples):
                    all_proba[i, predictions[i]] += weight

        # Reduce
        proba = all_proba / self.n_estimators

        return proba













