"""
MultipleSourceTrAdaBoostR2 algorithm

based on algorithm 3 in paper "Boosting for transfer learning with multiple sources".

"""

import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class MultipleSourceTrAdaBoostR2:
    def __init__(self,
                 base_estimator = DecisionTreeRegressor(max_depth=4),
                 sample_size = None,
                 n_estimators = 50,
                 learning_rate = 1.,
                 loss = 'linear',
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_sources = len(self.sample_size)-1
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state


    def fit(self, X, y, sample_weight=None):
        # Check parameters
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        if sample_weight is None:
            # Initialize weights to 1 / n_samples
            sample_weight = np.empty(X.shape[0], dtype=np.float64)
            sample_weight[:] = 1. / X.shape[0]
        else:
            # Normalize existing weights
            sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

            # Check that the sample weights sum is positive
            if sample_weight.sum() <= 0:
                raise ValueError(
                      "Attempting to fit with a non-positive "
                      "weighted number of samples.")

        if self.sample_size is None:
            raise ValueError("Additional input required: sample size of source and target is missing")
        elif np.array(self.sample_size).sum() != X.shape[0]:
            raise ValueError("Input error: the specified sample size does not equal to the input size")

        # Clear any previous fit results
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        self.accept_source = np.full(self.n_estimators, -1)

        for iboost in range(self.n_estimators): # this for loop is sequential and does not support parallel(revison is needed if making parallel)
            # Boosting step
            # 全てのsourceごとの弱学習器を初期化
            weak_estimators = []
            # 
            # sourceごとの合計予測誤差を格納する変数を初期化
            eachstep_estimator_error = np.ones(self.n_sources, dtype=np.float64)
            # sourceごとの予測誤差ベクトルを格納する変数を初期化
            error_vects = []
            # sourceごとに弱学習器作成
            before_sum_size = 0
            for idx, isource in enumerate(self.sample_size[:-1]):
                # 対象のsource以外のsourceデータを除く
                X_isource = np.concatenate([X[before_sum_size:before_sum_size+isource], X[-sample_size[-1]:]])
                y_isource = np.concatenate([y[before_sum_size:before_sum_size+isource], y[-sample_size[-1]:]])
                sample_weight_isource = np.concatenate([sample_weight[before_sum_size:before_sum_size+isource], sample_weight[-sample_size[-1]:]])
                # 弱学習器の作成と予測誤差の計算
                eachstep_estimator_error[idx], error_vect, weak_estimator = self._multipleSourceTrAdaBoostR2(
                        iboost,
                        X_isource, y_isource,
                        sample_weight_isource)
                weak_estimators.append(weak_estimator)
                error_vects.append(error_vect)

                before_sum_size += isource

            # 最も誤差の小さいsourceからの弱学習器を採用する
            self.accept_source[iboost] = np.argmin(eachstep_estimator_error)
            self.estimators_.append(weak_estimators[np.argmin(eachstep_estimator_error)])
            self.estimator_errors_[iboost] = eachstep_estimator_error[np.argmin(eachstep_estimator_error)]
            
            # 
            if self.estimator_errors_[iboost] <= 0:
                # Stop if fit is perfect
                print("fit is prefect")
                break

            # 予測誤差が0.5以上ならその弱学習器は不採用
            elif self.estimator_errors_[iboost] >= 0.5:
                # Discard current estimator only if it isn't the only one
                if len(self.estimators_) > 1:
                    self.estimators_.pop(-1)
                    # self.estimator_errors_[iboost] = 

            # 予測誤差が0<ε<0.5なら続行
            else:
                # αを計算
                alpha = 0.5 * np.log((1-self.estimator_errors_[iboost])/self.estimator_errors_[iboost])

            # sample_weight更新
            # target
            sample_weight[-self.sample_size[-1]:] *= np.exp(-alpha*error_vects[self.accept_source[iboost]][-self.sample_size[-1]:])
            # source
            accept_source_idx = sum(self.sample_size[:self.accept_source[iboost]])
            sample_weight[accept_source_idx:accept_source_idx+self.sample_size[self.accept_source[iboost]]] *= (
                np.exp(alpha*error_vects[self.accept_source[iboost]][:-self.sample_size[-1]])
            )

            # normalize
            sample_weight_sum = sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum
            
            # sample_weight, estimator_weight, estimator_error = self._multipleSourceTrAdaBoostR2(
            #         iboost,
            #         X, y,
            #         sample_weight)
            # # Early termination
            # if sample_weight is None:
            #     break

            # self.estimator_weights_[iboost] = estimator_weight
            # self.estimator_errors_[iboost] = estimator_error

            # # Stop if error is zero
            # if estimator_error == 0:
            #     break

            # sample_weight_sum = np.sum(sample_weight)

            # # Stop if the sum of sample weights has become non-positive
            # if sample_weight_sum <= 0:
            #     break

            # if iboost < self.n_estimators - 1:
            #     # Normalize
            #     sample_weight /= sample_weight_sum
        return self



    def _multipleSourceTrAdaBoostR2(self, iboost, X_isource, y_isource, sample_weight_isource):

        estimator = copy.deepcopy(self.base_estimator) # some estimators allow for specifying random_state estimator = base_estimator(random_state=random_state)

        ## using sampling method to account for sample_weight as discussed in Drucker's paper
        # Weighted sampling of the training set with replacement
        cdf = np.cumsum(sample_weight_isource)
        cdf /= cdf[-1]
        uniform_samples = self.random_state.random_sample(X_isource.shape[0])
        bootstrap_idx = cdf.searchsorted(uniform_samples, side='right')
        # searchsorted returns a scalar
        bootstrap_idx = np.array(bootstrap_idx, copy=False)

        # Fit on the bootstrapped sample and obtain a prediction
        # for all samples in the training set
        estimator.fit(X_isource[bootstrap_idx], y_isource[bootstrap_idx])
        y_predict = estimator.predict(X_isource)

        # self.estimators_.append(estimator)  # add the fitted estimator

        error_vect = np.abs(y_predict - y_isource)
        # あるsourceを使っての予測誤差ε
        estimator_error_isource = (
            sum(error_vect[-self.sample_size[-1]:] * sample_weight_isource[-self.sample_size[-1]:])
            / sum(sample_weight_isource[-self.sample_size[-1]:])
        )

        return estimator_error_isource, error_vect, estimator

        """
        if error_max != 0.:
            error_vect /= error_max

        if self.loss == 'square':
            error_vect **= 2
        elif self.loss == 'exponential':
            error_vect = 1. - np.exp(- error_vect)

        # Calculate the average loss
        estimator_error = (sample_weight * error_vect).sum()

        if estimator_error <= 0:
            # Stop if fit is perfect
            return sample_weight, 1., 0.

        elif estimator_error >= 0.5:
            # Discard current estimator only if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        beta = estimator_error / (1. - estimator_error)

        # avoid overflow of np.log(1. / beta)
        if beta < 1e-308:
            beta = 1e-308
        estimator_weight = self.learning_rate * np.log(1. / beta)

        # Boost weight using AdaBoost.R2 alg except the weight of the source data
        # the weight of the source data are remained
        source_weight_sum= np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
        target_weight_sum = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)

        if not iboost == self.n_estimators - 1:
            # targetのsample_weightを更新（当たっていれば重みを小さく、外れていれば重みを大きく）
            sample_weight[-self.sample_size[-1]:] *= np.power(
                    beta,
                    (1. - error_vect[-self.sample_size[-1]:]) * self.learning_rate)
            # sourceのsample_weight更新（当たっていれば重みを大きく、外れていれば重みを小さく）
            sample_weight[:-self.sample_size[-1]] *= np.power(
                    beta,
                    -(1. - error_vect[:-self.sample_size[-1]]) * self.learning_rate)
            # make the sum weight of the source data not changing
            source_weight_sum_new = np.sum(sample_weight[:-self.sample_size[-1]]) / np.sum(sample_weight)
            target_weight_sum_new = np.sum(sample_weight[-self.sample_size[-1]:]) / np.sum(sample_weight)
            if source_weight_sum_new != 0. and target_weight_sum_new != 0.:
                sample_weight[:-self.sample_size[-1]] = sample_weight[:-self.sample_size[-1]]*source_weight_sum/source_weight_sum_new
                sample_weight[-self.sample_size[-1]:] = sample_weight[-self.sample_size[-1]:]*target_weight_sum/target_weight_sum_new

        return sample_weight, estimator_weight, estimator_error
        """


    def predict(self, X):
        # Evaluate predictions of all estimators
        predictions = np.array([
                est.predict(X) for est in self.estimators_[:len(self.estimators_)]]).T

        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)

        # Find index of median prediction for each sample
        weight_cdf = np.cumsum(self.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]

        # Return median predictions
        return predictions[np.arange(X.shape[0]), median_estimators]


if __name__ == "__main__":
    sample_size = [50, 100, 75, 15]
    X = np.random.rand(sum(sample_size), 5)
    y = np.random.randint(0, 2, size=sum(sample_size))
    clf = MultipleSourceTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                            n_estimators = 100, sample_size = sample_size, 
                            random_state = np.random.RandomState(1))
    clf.fit(X, y)