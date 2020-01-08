"""
MultipleSourceTrAdaBoost algorithm

based on algorithm 3 in paper "Boosting for transfer learning with multiple sources".

"""

import numpy as np
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


class MultipleSourceTrAdaBoost:
    def __init__(self,
                 base_estimator = DecisionTreeClassifier(max_depth=4),
                 sample_size = None,
                 n_estimators = 50,
                 learning_rate = 1.,
                 random_state = np.random.mtrand._rand):
        self.base_estimator = base_estimator
        self.sample_size = sample_size
        self.n_sources = len(self.sample_size)-1
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
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
        estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        accept_source = np.full(self.n_estimators, -1)

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
                X_isource = np.concatenate([X[before_sum_size:before_sum_size+isource], X[-self.sample_size[-1]:]])
                y_isource = np.concatenate([y[before_sum_size:before_sum_size+isource], y[-self.sample_size[-1]:]])
                sample_weight_isource = np.concatenate([sample_weight[before_sum_size:before_sum_size+isource], sample_weight[-self.sample_size[-1]:]])
                # 弱学習器の作成と予測誤差の計算
                eachstep_estimator_error[idx], error_vect, weak_estimator = self._MultipleSourceTrAdaBoost(
                        iboost,
                        X_isource, y_isource,
                        sample_weight_isource)
                weak_estimators.append(weak_estimator)
                error_vects.append(error_vect)

                before_sum_size += isource

            # 最も誤差の小さいsourceからの弱学習器を採用する
            accept_source[iboost] = np.argmin(eachstep_estimator_error)
            self.estimators_.append(weak_estimators[np.argmin(eachstep_estimator_error)])
            estimator_errors_[iboost] = eachstep_estimator_error[np.argmin(eachstep_estimator_error)]
            
            # 完璧に予測出来たら終了
            if estimator_errors_[iboost] <= 0:
                # Stop if fit is perfect
                print("fit is prefect")
                break

            # 予測誤差が0.5以上ならその弱学習器は不採用&終了
            elif estimator_errors_[iboost] >= 0.5:
                # Discard current estimator only if it isn't the only one
                if len(self.estimators_) > 1:
                    self.estimators_.pop(-1)
                    # estimator_errors_[iboost] = -1
                break


            # 予測誤差が0<ε<0.5なら続行
            # αを計算
            alpha = 0.5 * np.log((1-estimator_errors_[iboost])/estimator_errors_[iboost])
            # avoid overflow of np.log(1. / alpha)
            if alpha < 1e-308:
                alpha = 1e-308
            estimator_weights_[iboost] = self.learning_rate * alpha

            # sample_weight更新
            # target
            sample_weight[-self.sample_size[-1]:] *= np.exp(-alpha*error_vects[accept_source[iboost]][-self.sample_size[-1]:])
            # source
            accept_source_idx = sum(self.sample_size[:accept_source[iboost]])
            sample_weight[accept_source_idx:accept_source_idx+self.sample_size[accept_source[iboost]]] *= (
                np.exp(alpha*error_vects[accept_source[iboost]][:-self.sample_size[-1]])
            )

            # normalize
            sample_weight_sum = sum(sample_weight)

            # Stop if the sum of sample weights has become non-positive
            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators - 1:
                # Normalize
                sample_weight /= sample_weight_sum

        # ループが途中で止まった場合、その途中までの結果のみ取り出す
        # 反映されない部分はestimator_errors_[i]>=0.5(=0で止まったときはそのステップまで採用するからこの条件でok)
        self.estimator_weights_ = estimator_weights_[estimator_errors_<0.5]
        self.accept_source = accept_source[estimator_errors_<0.5]
        self.estimator_errors_ = estimator_errors_[estimator_errors_<0.5]
        return self



    def _MultipleSourceTrAdaBoost(self, iboost, X_isource, y_isource, sample_weight_isource):

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
    y1 = np.where(np.sum(X, axis=1)>2, 1,0)
    y = np.random.randint(0, 2, size=sum(sample_size))
    clf = MultipleSourceTrAdaBoost(DecisionTreeClassifier(max_depth=6),
                            n_estimators = 100, sample_size = sample_size, 
                            random_state = np.random.RandomState(1))
    clf.fit(X, y1)
    clf.predict(X)