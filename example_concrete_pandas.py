"""
concrete compressiveを用いて実験

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 # 既存研究
from SimpleTrAdaBoostR2 import SimpleTrAdaBoostR2 # シンプルな改良版(分類問題と同じアルゴリズム)
from SimpleTrAdaBoostRT import SimpleTrAdaBoostRT # .RTを用いた改良版
# from MultipleSourceTrAdaBoostR2 import MultipleSourceTrAdaBoostR2 # 複数ソースに対応したSimpleTrAdaBoostR2
from MultipleSourceTrAdaBoostR2ver2 import MultipleSourceTrAdaBoostR2

# 前処理
y_feature = "Concrete compressive strength(MPa, megapascals) "
split_feature = "Superplasticizer (component 5)(kg in a m^3 mixture)"

# trainのデータサイズ
n_target = 15

# モデルパラメータ
n_estimators = 100
steps = 10
fold = 5
random_state = np.random.RandomState(1)



def make_dataset():
    df = pd.read_csv("./dataset/Concrete_Data.csv")
    domain_df = _split_dataset(df, split_feature)

    return domain_df
    


def _split_dataset(df, split_feature):
    df_sorted = df.sort_values(split_feature).reset_index(drop=True)
    df_sorted.drop(columns=split_feature, inplace=True)
    low = df_sorted[:len(df_sorted)//3].assign(Domain="low")
    medium = df_sorted[len(df_sorted)//3: (len(df_sorted)//3)*2].assign(Domain="medium")
    high = df_sorted[(len(df_sorted)//3)*2:].assign(Domain="high")

    domain_df = pd.concat([low, medium, high])
    
    return domain_df.sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    domain_df = make_dataset()
    msTrAda = MultipleSourceTrAdaBoostR2(target_label="high")
    msTrAda.fit(domain_df.drop(y_feature, axis=1).values, domain_df[y_feature].values)


"""
def experiment_main(count, idx_target, experiment_result):
    # データ取得
    _, low, medium, high = make_dataset()

    # 実験
    df_list = [low, medium, high]
    name_list = ["low", "medium", "high"]
    for idx, target_df_origin in enumerate(df_list):
        # データの分割
        # targetデータはデータ数を小さくするためランダムサンプリング
        target_df = target_df_origin.sample(n=n_target)
        target_X_train = target_df.drop(columns=y_feature).values
        target_Y_train = target_df[y_feature].values
        # サンプリングされなかったデータをtestデータとする
        target_df_test = target_df_origin.drop(index=target_df.index)
        target_X_test = target_df_test.drop(columns=y_feature).values
        target_Y_test = target_df_test[y_feature].values
        
        source_idx = list({0,1,2}^{idx})
        source1_X = df_list[source_idx[0]].drop(columns=y_feature).values
        source1_Y = df_list[source_idx[0]][y_feature].values
        source2_X = df_list[source_idx[1]].drop(columns=y_feature).values
        source2_Y = df_list[source_idx[1]][y_feature].values
        source_X = np.concatenate([source1_X, source2_X])
        source_Y = np.concatenate([source1_Y, source2_Y])

        X = np.concatenate([source_X, target_X_train])
        Y = np.concatenate([source_Y, target_Y_train])

        sample_size = [source_X.shape[0], target_X_train.shape[0]]

        # TargetとSourceの表示
        print("target : {:6}   source : {:6} {}"
            .format(name_list[idx], 
                name_list[source_idx[0]], 
                name_list[source_idx[1]]))


        # modelへの適用

        # 4.2 TwoStageAdaBoostR2
        regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                            n_estimators = n_estimators, sample_size = sample_size, 
                            steps = steps, fold = fold, 
                            random_state = random_state,
                            binary_search_step=1e-30)
        regr_1.fit(X, Y)
        y_pred1 = regr_1.predict(target_X_test)


        # Sourceデータもtargetデータと同じように重みを更新するシンプルなTrAdaBoostR2
        regr_2 = SimpleTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                            n_estimators = n_estimators, sample_size = sample_size, 
                            random_state = random_state)
        regr_2.fit(X, Y)
        y_pred2 = regr_2.predict(target_X_test)


        # sourceデータ1つ1つの誤差が閾値以下なら誤差を0にするTrAdaBoostRT
        regr_3 = SimpleTrAdaBoostRT(DecisionTreeRegressor(max_depth=6),
                            n_estimators = n_estimators, sample_size = sample_size, 
                            random_state = random_state)
        regr_3.fit(X, Y)
        y_pred3 = regr_3.predict(target_X_test)


        # MultipleSourceTrAdaBoost.R2
        regr_4 = MultipleSourceTrAdaBoostR2(DecisionTreeRegressor(max_depth=6),
                            n_estimators = n_estimators, sample_size = sample_size, 
                            random_state = random_state)
        regr_4.fit(X, Y)
        y_pred4 = regr_4.predict(target_X_test)

        # 4.3 As comparision, use AdaBoostR2 without transfer learning
        #==============================================================================
        regr_0 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
                                n_estimators = n_estimators)
        #==============================================================================
        regr_0.fit(target_X_train, target_Y_train)
        y_pred0 = regr_0.predict(target_X_test)

        # 4.4 Calculate mse
        mse_twostageboost = mean_squared_error(target_Y_test, y_pred1)   
        mse_simpleboost = mean_squared_error(target_Y_test, y_pred2)
        mse_simpleboostRT = mean_squared_error(target_Y_test, y_pred3)
        mse_multipleboost = mean_squared_error(target_Y_test, y_pred4)
        mse_adaboost = mean_squared_error(target_Y_test, y_pred0)
        print("MSE of regular AdaboostR2:", mse_adaboost)
        print("MSE of TwoStageTrAdaboostR2:", mse_twostageboost)
        print("MSE of SimpleTrAdaboostR2:", mse_simpleboost)
        print("MSE of SimpleTrAdaboostRT:", mse_simpleboostRT)
        print("MSE of multipleTrAdaboostR2:", mse_multipleboost)
        print("--------------------------------")


        # 結果の保存
        idx_target[count*3+idx] = name_list[idx]
        experiment_result.append([mse_adaboost, mse_twostageboost,
                                    mse_simpleboost, mse_simpleboostRT, mse_multipleboost])

    return idx_target, experiment_result
    

if __name__ == "__main__":
    # 実験
    idx_target = {}
    experiment_result = []
    for count in range(20):
        print("--------------------------------")
        print("experiment : ", count)
        print("--------------------------------")
        idx_target, experiment_result = experiment_main(count, idx_target, experiment_result)
    
    
    df_result = pd.DataFrame(experiment_result, columns=["AdaBoost", "Two-StageTrAdaBoost", "SimpleTrAdaBoostR2", "SimpleTrAdaBoostRT", "MultipleTrAdaBoostR2"])
    experiment_n = len(df_result)
    df_result.to_csv("./result/concrete_compressive_mse_n%s.csv" %experiment_n)

    # 実験結果の図
    result_mean = df_result.mean(axis=0)
    result_std = df_result.std(axis=0)
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(result_mean)), result_mean, yerr=result_std, tick_label=result_mean.index, ecolor="black", width=0.5)
    plt.title("Concrete Compressive MSE  n=%s" %len(df_result))
    plt.savefig("./result/concrete_compressive_mse_compare_n%s.png" %experiment_n)

    # targetごとの結果の違い
    df_result["target"] = df_result.reset_index().iloc[:,0].apply(lambda x: idx_target[x])
    for target_group in ["low", "medium", "high"]:
        df_target = df_result[df_result["target"]==target_group]
        result_mean = df_target.mean(axis=0)
        result_std = df_target.std(axis=0)
        plt.figure(figsize=(10,5))
        plt.bar(np.arange(len(result_mean)), result_mean, yerr=result_std, tick_label=result_mean.index, ecolor="black", width=0.5)
        plt.title("Concrete Compressive MSE (%s)  n=%s" %(target_group, len(df_target)))
        experiment_n = len(df_target)
        plt.savefig("./result/concrete_compressive_mse_compare_%s_n%s.png" %(target_group, experiment_n))

"""