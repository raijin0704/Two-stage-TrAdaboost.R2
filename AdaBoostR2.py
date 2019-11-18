from abc import ABCMeta, abstractmethod

from sklearn.base import RegressorMixin, BaseEnsemble

class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_params=tuple(),
                 learning_rate=1.,
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params)

        self.learning_rate = learning_rate
        self.random_state = random_state

    def _validate_data(self, X, y=None):

        # Accept or convert to these sparse matrix formats so we can
        # use _safe_indexing
        accept_sparse = ['csr', 'csc']
        if y is None:
            ret = check_array(X,
                              accept_sparse=accept_sparse,
                              ensure_2d=False,
                              allow_nd=True,
                              dtype=None)
        else:
            ret = check_X_y(X, y,
                            accept_sparse=accept_sparse,
                            ensure_2d=False,
                            allow_nd=True,
                            dtype=None,
                            y_numeric=is_regressor(self))
        return ret

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier/regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
        """
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        X, y = self._validate_data(X, y)

        sample_weight = _check_sample_weight(sample_weight, X, np.float64)
        sample_weight /= sample_weight.sum()
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight cannot contain negative weights")

        # Check parameters
        self._validate_estimator()

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        for iboost in range(self.n_estimators):
            # Boosting step
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost,
                X, y,
                sample_weight,
                random_state)

            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            # Stop if error is zero
            if estimator_error == 0:
                break

            sample_weight_sum = np.sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        return self

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.
        Warning: This method needs to be overridden by subclasses.
        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        sample_weight : array-like of shape (n_samples,)
            The current sample weights.
        random_state : RandomState
            The current random number generator
        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.
        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.
        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.
        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            Labels for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Yields
        ------
        z : float
        """
        X = self._validate_data(X)

        for y_pred in self.staged_predict(X):
            if is_classifier(self):
                yield accuracy_score(y, y_pred, sample_weight=sample_weight)
            else:
                yield r2_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self):
        """Return the feature importances (the higher, the more important the
           feature).
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The feature importances.
        """
        if self.estimators_ is None or len(self.estimators_) == 0:
            raise ValueError("Estimator not fitted, "
                             "call `fit` before `feature_importances_`.")

        try:
            norm = self.estimator_weights_.sum()
            return (sum(weight * clf.feature_importances_ for weight, clf
                    in zip(self.estimator_weights_, self.estimators_))
                    / norm)

        except AttributeError:
            raise AttributeError(
                "Unable to compute feature importances "
                "since base_estimator does not have a "
                "feature_importances_ attribute")


class AdaBoostRegressor(RegressorMixin, BaseWeightBoosting):
    """An AdaBoost regressor.
    An AdaBoost [1] regressor is a meta-estimator that begins by fitting a
    regressor on the original dataset and then fits additional copies of the
    regressor on the same dataset but where the weights of instances are
    adjusted according to the error of the current prediction. As such,
    subsequent regressors focus more on difficult cases.
    This class implements the algorithm known as AdaBoost.R2 [2].
    Read more in the :ref:`User Guide <adaboost>`.
    .. versionadded:: 0.14
    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        If ``None``, then the base estimator is
        ``DecisionTreeRegressor(max_depth=3)``.
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
    learning_rate : float, optional (default=1.)
        Learning rate shrinks the contribution of each regressor by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.
    loss : {'linear', 'square', 'exponential'}, optional (default='linear')
        The loss function to use when updating the weights after each
        boosting iteration.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of classifiers
        The collection of fitted sub-estimators.
    estimator_weights_ : array of floats
        Weights for each estimator in the boosted ensemble.
    estimator_errors_ : array of floats
        Regression error for each estimator in the boosted ensemble.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances if supported by the ``base_estimator``.
    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    >>> regr.fit(X, y)
    AdaBoostRegressor(n_estimators=100, random_state=0)
    >>> regr.feature_importances_
    array([0.2788..., 0.7109..., 0.0065..., 0.0036...])
    >>> regr.predict([[0, 0, 0, 0]])
    array([4.7972...])
    >>> regr.score(X, y)
    0.9771...
    See also
    --------
    AdaBoostClassifier, GradientBoostingRegressor,
    sklearn.tree.DecisionTreeRegressor
    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.
    .. [2] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.
    """
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 loss='linear',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Build a boosted regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        y : array-like of shape (n_samples,)
            The target values (real numbers).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.
        Returns
        -------
        self : object
        """
        # Check loss
        if self.loss not in ('linear', 'square', 'exponential'):
            raise ValueError(
                "loss must be 'linear', 'square', or 'exponential'")

        # Fit
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super()._validate_estimator(
            default=DecisionTreeRegressor(max_depth=3))

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost for regression
        Perform a single boost according to the AdaBoost.R2 algorithm and
        return the updated sample weights.
        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,)
            The current sample weights.
        random_state : RandomState
            The current random number generator
        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.
        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.
        estimator_error : float
            The regression error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        # Weighted sampling of the training set with replacement
        bootstrap_idx = random_state.choice(
            np.arange(_num_samples(X)), size=_num_samples(X), replace=True,
            p=sample_weight
        )

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        X_ = _safe_indexing(X, bootstrap_idx)
        y_ = _safe_indexing(y, bootstrap_idx)
        estimator.fit(X_, y_)
        y_predict = estimator.predict(X)

        error_vect = np.abs(y_predict - y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]

        error_max = masked_error_vector.max()
        if error_max != 0:
            masked_error_vector /= error_max

        if self.loss == 'square':
            masked_error_vector **= 2
        elif self.loss == 'exponential':
            masked_error_vector = 1. - np.exp(-masked_error_vector)

        # Calculate the average loss
        estimator_error = (masked_sample_weight * masked_error_vector).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # Boost weight using AdaBoost.R2 alg
        estimator_weight = self.learning_rate * np.log(1. / beta)

        if not iboost == self.n_estimators - 1:
            sample_weight[sample_mask] *= np.power(
                beta, (1. - masked_error_vector) * self.learning_rate
            )

        return sample_weight, estimator_weight, estimator_error

    def _get_median_predict(self, X, limit):
        # Evaluate predictions of all estimators
        predictions = np.array([
            est.predict(X) for est in self.estimators_[:limit]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]

        # Return median predictions
        return predictions[np.arange(_num_samples(X)), median_estimators]

    def predict(self, X):
        """Predict regression value for X.
        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        return self._get_median_predict(X, len(self.estimators_))

    def staged_predict(self, X):
        """Return staged predictions for X.
        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.
        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        Yields
        -------
        y : generator of array, shape = [n_samples]
            The predicted regression values.
        """
        check_is_fitted(self)
        X = self._validate_data(X)

        for i, _ in enumerate(self.estimators_, 1):
            yield self._get_median_predict(X, limit=i)