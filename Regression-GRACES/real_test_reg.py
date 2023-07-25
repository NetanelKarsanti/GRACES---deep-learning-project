import os
from GRACES import GRACES
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
from sklearn import svm
import warnings
import pandas as pd
import joblib
import torch
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error, mean_squared_error


warnings.filterwarnings("ignore")
dropout_prob = [0.1, 0.5, 0.9]
f_correct = [0, 0.1, 0.5, 0.9]


def combin_channel(x_train, y_train,x_val,y_val, selection_g, futers_map):
    chnels_dupl = []
    for feature in selection_g:
        for key, value in futers_map.items():
            if feature in value:
                chnels_dupl.append(key)
                break
    chnels = list(set(chnels_dupl))
    future_index = []
    for key in chnels:
        if key in futers_map:
            future_index.extend(futers_map[key])

    all_x_train = x_train[:,future_index]
    all_x_val = x_val[:,future_index]
    if all_x_train.shape[1]>1 :
        pipeline = Pipeline([('select', SelectKBest(score_func=f_regression)),('model', RandomForestRegressor())])
        param_grid = {'select__k': range(0, all_x_train.shape[1]-4)}
        grid_search = GridSearchCV(pipeline, param_grid, cv=3)
        grid_search.fit(all_x_train, y_train[:,0])
        best_k = grid_search.best_params_['select__k']
        best_selector = SelectKBest(score_func=f_regression, k=best_k)
        X_best = best_selector.fit_transform(all_x_train, y_train[:,0])
    else:
        X_best = all_x_train

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_best, y_train[:, 0])
    feature_indices = best_selector.get_support(indices=True)

    feature_s = [future_index[x] for x in feature_indices.tolist()]

    chnele_chose = []
    for feature in feature_s:
        for key, value in futers_map.items():
            if feature in value:
                chnele_chose.append(key)
                break
    chnele_chose = list(set(chnele_chose))

    y_pred = rf_regressor.predict(all_x_val[:,feature_indices])
    mse = mean_squared_error(y_val[:, 1], y_pred)
    mae = mean_absolute_error(y_val[:, 1], y_pred)
    std = np.std((y_val[:, 1], y_pred))
    chnel_rep = {'input_chanel':chnels, 'best_comb_chanel' : chnele_chose}
    return mse, mae, std, chnel_rep

def main(name, n_features, n_iters, n_repeats, futers_map,seed_):
    np.random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_)
        torch.cuda.manual_seed_all(seed_)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = scipy.io.loadmat('data/' +name)
    x = data['X'].astype(float)
    y = data['Y']
    y = y.astype(float)
    mse_per_iter = []
    selection_per_iter = []
    mae_per_iter =[]
    std_per_iter =[]
    all_chnel_per_iter =[]
    chose_chanel_per_iter=[]
    GNN_Reliability_per_iter = []

    seeds = np.random.choice(range(100), n_iters, replace=False)
    for iter in tqdm(range(n_iters)):

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=seeds[iter], stratify=y[:,1])
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=2/7, random_state=seeds[iter])
        loss_grid = np.zeros((len(dropout_prob), len(f_correct)))
        slc_test = GRACES(n_features=n_features)
        score = slc_test.mutual_test(x_train, y_train)
        for i in range(len(dropout_prob)):
            for j in range(len(f_correct)):
                for r in range(n_repeats):
                    slc_g = GRACES(n_features=n_features, dropout_prob=dropout_prob[i], f_correct=f_correct[j])
                    selection_g = slc_g.select(x_train, y_train, score)
                    mse, mae, std, chnels = combin_channel(x_train, y_train, x_val,y_val, selection_g, futers_map)
                    loss_grid[i, j] += mae + std
        index_i, index_j = np.where(loss_grid == np.min(loss_grid))
        best_index = np.argmin(loss_grid[index_i, index_j])
        best_prob, best_f_correct = dropout_prob[int(index_i[best_index])], f_correct[int(index_j[best_index])]
        MSE_per_repeat = []
        MAE_per_repeat = []
        Std_per_repeat = []
        all_score = []
        chanel_rep = []
        min_chanel_rep = []
        for r in range(n_repeats):
            slc = GRACES(n_features=n_features, dropout_prob=best_prob, f_correct=best_f_correct)
            selection = slc.select(x_train, y_train, score)
            mse_test, mae_test, std_test,chnels_inf = combin_channel(x_train, y_train, x_test, y_test, selection, futers_map)
            all_score.append(mse_test + mae_test + std_test)
            MSE_per_repeat.append(mse_test); MAE_per_repeat.append(mae_test); Std_per_repeat.append(std_test)
            chanel_rep.append(chnels_inf['input_chanel'])
            min_chanel_rep.append(chnels_inf['best_comb_chanel'])

        a = [len(x) for x in min_chanel_rep]
        min_index_per_repeat = a.index(min(a))
        best_mse_per_repeat = MSE_per_repeat[min_index_per_repeat]
        best_mae_per_repeat = MAE_per_repeat[min_index_per_repeat]
        best_Std_per_repeat = Std_per_repeat[min_index_per_repeat]
        all_best_chnel_per_repeat = chanel_rep[min_index_per_repeat]
        chose_best_chnel_per_repeat = min_chanel_rep[min_index_per_repeat]

        mse_per_iter.append(best_mse_per_repeat)
        mae_per_iter.append(best_mae_per_repeat)
        std_per_iter.append(best_Std_per_repeat)
        all_chnel_per_iter.append(all_best_chnel_per_repeat)
        chose_chanel_per_iter.append(chose_best_chnel_per_repeat)
        GNN_Reliability_per_iter.append(index_j.item())


    min_C = [len(x) for x in chose_chanel_per_iter]
    min_C_index= min_C.index(min(min_C))
    best_mse_per_iter = mse_per_iter[min_C_index]
    best_mae_per_iter = mae_per_iter[min_C_index]
    best_std_per_iter = std_per_iter[min_C_index]
    all_chan_comb = all_chnel_per_iter[min_C_index]
    chose_chanel= chose_chanel_per_iter[min_C_index]
    GNN_Reliability = f_correct[GNN_Reliability_per_iter[min_C_index]]

    result = {'MSE':best_mse_per_iter, 'MAE':best_mae_per_iter, 'STD':best_std_per_iter, 'ALL_FEATURE':all_chan_comb,
              'Best_Chanel_comb':chose_chanel, 'Reliability of GNN': GNN_Reliability }
    return result


if __name__ == "__main__":
    name = 'Div2_MinMax_N564'
    _map = joblib.load(r'data/futers_dic.joblib')
    max_features = 50
    n_iters = 2
    n_repeats = 2
    save = True
    for p in range(max_features):
        result = main(name=name, n_features=p + 1, n_iters=n_iters, n_repeats=n_repeats, futers_map=_map, seed_=42)
        print('=========================== '+str(p+1)+' : Features'+' ==============================')
        print(' MSE                                          : ' + str(result['MSE']))
        print(' MAE                                          : ' + str(result['MAE']))
        print(' STD                                          : ' + str(result['STD']))
        print(' All Channels from Feature selections process : ' + str(result['ALL_FEATURE']))
        print(' Best Channels combination                    : ' + str(result['Best_Chanel_comb']))
        print(' Reliability of GNN                           : ' + str(result['Reliability of GNN']*100) + '%')
        print('========================================================================')

        if save:
            os.chdir('/home/ARO.local/netanelk/Projects/Featuer_selection_res/Normal_by_564/')
            joblib.dump(result,'Process_for_'+str(p+1)+'features.joblib')
            os.chdir('/home/ARO.local/netanelk/Projects/Feature_selection_reg/')