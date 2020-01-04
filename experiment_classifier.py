import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# from TrAdaBoost import TrAdaBoostClassifier
from MultipleSourceTrAdaBoost import MultipleSourceTrAdaBoost

if __name__ == "__main__":
    sample_size = [50, 100, 75, 15]
    X = np.random.rand(sum(sample_size), 5)
    # X_domain =

    # 完璧に分類してほしいやつ
    y1 = np.where(np.sum(X, axis=1)>2, 1,0)
    # 分類できないやつ
    y = np.random.randint(0, 2, size=sum(sample_size))

    adaboost = AdaBoostClassifier()
    adaboost.fit(X[-15:], y1[-15:])
    adaboost_predict = adaboost.predict(X[:-15])
    print(np.count_nonzero(y1[:-15]==adaboost_predict)/len(adaboost_predict))

    # tradaboost = TrAdaBoostClassifier()
    clf = MultipleSourceTrAdaBoost(DecisionTreeClassifier(max_depth=6),
                            n_estimators = 100, sample_size = sample_size, 
                            random_state = np.random.RandomState(1))

    clf.fit(X, y1)
    multi_predict = clf.predict(X[:-15])
    print(np.count_nonzero(y1[:-15]==multi_predict)/len(adaboost_predict))