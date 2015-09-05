"""CCEL estimator."""

from __future__ import division

import numbers
import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.base import ClassifierMixin
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.externals.six import with_metaclass
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state, check_X_y, check_array, column_or_1d
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

from sklearn.ensemble.base import BaseEnsemble


__all__ = ["CCELClassifier"]

MAX_INT = np.iinfo(np.int32).max

class Organizm:
    def __init__(self, n_samples, n_features, ps, pf, base_estimator, random_state):
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

        self.cache_estimator = base_estimator
        self.cache_est_weight = 1
        self.cache_accuracy = 0
        self.cache_contribution = 0
        self.cache_predictions = None
        self.first = None
        self.second = None

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
        self.first = first
        self.second = second

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

"""class Population():
    def __init__(self, N0, N1, N2, ):
"""

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
        assert N2 < N0, "N2 must be less than size of initial population"
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

    def fit(self, X, y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
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
        self.estimators_features_ = []
        self.estimators_weights_ = []

        accuracy_per_generation = np.zeros((self.tmax,), dtype=float)
        _best_accuracy = 0
        _populations = []
        _contributions = []
        _offsprings = [Organizm(n_samples, self.n_features_, self.ps, self.pf,
                                self._make_estimator(append=False),
                                random_state) for _ in range(self.N1)]

        def _append_population():
            idx = len(_populations)
            population = [Organizm(n_samples, self.n_features_, self.ps, self.pf,
                                   self._make_estimator(append=False),
                                   random_state) for _ in range(self.N0)]
            _populations.append(population)
            _contributions.append(np.arange(self.d1, dtype=float))
            _compute_cache_predictions(population)
            _compute_cache_accuracy_contribution(idx, population)

        def _remove_population(idx):
            del _populations[idx]
            del _contributions[idx]

        def _compute_cache_accuracy_contribution(exclude_idx, target_population):
            #_score(self, Y, n_estimators, sample_weights=None, est_weights=None):
            Y=0
            if(len(_populations) > 1):
                for idx, population in enumerate(_populations):
                    if idx != exclude_idx:
                        Y += population[0].cache_est_weight*population[0].cache_predictions
                accuracy_without_target = self._score(y, Y, sample_weight)
            else:
                accuracy_without_target = 0

            for o in target_population:
                o.cache_accuracy = self._score(y, Y+(o.cache_est_weight*o.cache_predictions),
                                               sample_weight)
                # need to rework
                o.cache_contribution = np.abs(o.cache_accuracy-accuracy_without_target)

        def _compute_cache_predictions(target_population):
            for o in target_population:
                if support_sample_weight:
                    o.cache_estimator.fit(X[o.genome_samples,:][:,o.genome_features],
                                          y[o.genome_samples], sample_weight)
                else:
                    o.cache_estimator.fit(X[o.genome_samples,:][:,o.genome_features],
                                          y[o.genome_samples])
                if support_predict_proba:
                    o.cache_predictions = o.cache_estimator.predict_proba(X[:, o.genome_features])
                else:
                    o.cache_predictions = np.zeros((n_samples, len(o.cache_estimator.classes_)))
                    predictions = o.cache_estimator.predict(X[:, o.genome_features])
                    for i in range(n_samples):
                        o.cache_predictions[i, predictions[i]] += 1

        def genchoices():
            res = set()
            while len(res) <= self.N1:
                first = random_state.randint(0, self.N0-1)
                second = random_state.randint(first+1, self.N0)
                res.add((first, second))
            return res

        # Initialize first population
        for i in range(self.min_estimators):
            _append_population()

        for generation in range(self.tmax):
            remove = []
            for idx in range(len(_populations)):
                population = _populations[idx]
                # crossover
                for offspring, (first, second) in zip(_offsprings, genchoices()):
                    offspring.crossover(population[first], population[second])
                    offspring.mutation(self.pm)

                _compute_cache_accuracy_contribution(idx, population)
                _compute_cache_predictions(_offsprings)
                _compute_cache_accuracy_contribution(idx, _offsprings)
                population.sort(reverse=True, key=lambda x:x.cache_accuracy)
                a = sorted(population[:self.N2] + _offsprings, reverse=True, key=lambda x:x.cache_accuracy)
                dead = population[self.N2:]
                _populations[idx] = a[:self.N0]
                _offsprings = dead+a[self.N0:]
                accuracy_per_generation[generation] = max(accuracy_per_generation[generation],
                                                             a[0].cache_accuracy)
                print("Estimator #{0} from {1}, accuracy {2}, contribution {3}".format(idx, len(_populations), a[0].cache_accuracy, a[0].cache_contribution))
                _contributions[idx][generation%self.d1] = a[0].cache_contribution

                if len(_populations) > self.min_estimators and np.abs(_contributions[idx].mean()) < self.eps1:
                    print("Estimator #{0} removed, contribution was {1}".format(idx, np.abs(_contributions[idx][generation%self.d1].mean())))
                    remove.append(idx)

            """for r in sorted(remove, reverse=True):
                _remove_population(r)
            if generation-self.d2 >=0 and (accuracy_per_generation[generation-self.d2:generation+1].max() -
                accuracy_per_generation[generation-self.d2:generation+1].min()) < self.eps2:
                _append_population()
                print("Stagnation, let's add new population")

            if generation-self.d3 >=0 and (accuracy_per_generation[generation-self.d3:generation+1].max() -
                                            accuracy_per_generation[generation-self.d3:generation+1].min()) < self.eps3:
                print("Seems that adding new population doesn't helps, stopping...")
                break
            """

        for population in _populations:
            self.estimators_ = [population[0].estimator]
            self.estimators_features_ = [population[0].genome_features > 0]
            self.estimators_weights_ = [population[0].cache_est_weight]

        return self

    def _validate_y(self, y):
        # Default implementation
        return column_or_1d(y, warn=True)

    @abstractmethod
    def _score(self, y_true, Y, sample_weight=None):
        """Returns score from precomputed results of each estimator"""


class CCELClassifier(BaseCCEL, ClassifierMixin):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : object or None, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    bootstrap_features : boolean, optional (default=False)
        Whether features are drawn with replacement.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

    n_jobs : int, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    base_estimator_ : list of estimators
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : array of shape = [n_classes]
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    """
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
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of a an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
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

                if self.n_classes == len(estimator.classes_):
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
        proba = sum(all_proba) / self.n_estimators

        return proba

    def _score(self, y_true, Y,  sample_weight=None):
        ensemble_y = self.classes_.take((np.argmax(Y, axis=1)),
                                      axis=0)
        return accuracy_score(y_true, ensemble_y, sample_weight=sample_weight, normalize=True)

if __name__ == "__main__":
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    #y = np.array([0, 1, 1, 0])

    """
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
                 warm_start=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):"""
    from sklearn import datasets
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    data = datasets.load_digits()

    b = BaggingClassifier(DecisionTreeClassifier(max_depth=7), n_estimators=100)
    b.fit(data.data, data.target)
    print (b.score(data.data, data.target))
    # TODO min_features and min_samples control
    # combinations in N1 control
    # proper threating of classifiers without weighting and proba_est
    # weighting
    # stopping criterion
    c = CCELClassifier(tmax=150, po=0.3, pf=0.07, pm=0.2, N2=5, N1=100, N0=20, d1=10, eps1=0.05,
                       d2=15, eps2=0.07, d3=30, eps3=0.02, min_features=2, min_samples=0.5, min_estimators=25,
                       base_estimator=GaussianNB())

    c.fit(data.data, data.target)

















