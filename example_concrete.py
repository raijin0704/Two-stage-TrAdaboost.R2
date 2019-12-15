"""
concrete compressiveを用いて実験

"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2 # 既存研究
from SimpleTrAdaBoostR2 import SimpleTrAdaBoostR2 # シンプルな改良版(分類問題と同じアルゴリズム)
from SimpleTrAdaBoostRT import SimpleTrAdaBoostRT # .RTを用いた改良版


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
    df = pd.read_csv("./dataset/concrete_compressive/Concrete_Data.csv")
    low, medium, high = _split_dataset(df, split_feature)

    return df, low, medium, high
    


def _split_dataset(df, split_feature):
    df_sorted = df.sort_values(split_feature).reset_index(drop=True)
    df_sorted.drop(columns=split_feature, inplace=True)
    low = df_sorted[:len(df_sorted)//3]
    medium = df_sorted[len(df_sorted)//3: (len(df_sorted)//3)*2]
    high = df_sorted[(len(df_sorted)//3)*2:]
    
    return low, medium, high


def experiment_main(count):
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
                            random_state = random_state)
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
        mse_adaboost = mean_squared_error(target_Y_test, y_pred0)
        print("MSE of regular AdaboostR2:", mse_adaboost)
        print("MSE of TwoStageTrAdaboostR2:", mse_twostageboost)
        print("MSE of SimpleTrAdaboostR2:", mse_simpleboost)
        print("MSE of SimpleTrAdaboostRT:", mse_simpleboostRT)
        print("--------------------------------")



if __name__ == "__main__":
    for count in range(1,6):
        print("--------------------------------")
        print("experiment : ", count)
        print("--------------------------------")
        experiment_main(count)
