import os
import joblib
from GRACES import GRACES
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn import svm
import torch
import warnings

warnings.filterwarnings("ignore")

dropout_prob = [0.1, 0.5, 0.75]
f_correct = [0, 0.1, 0.5, 0.9]

def main(name, n_features, n_iters, n_repeats):
    seed_ = 42
    np.random.seed(0)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed_)
        torch.cuda.manual_seed_all(seed_)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = scipy.io.loadmat('data/' + name)
    x = data['X'].astype(float)
    y = data['Y'][:,1] # to multy class problem
    seeds = np.random.choice(range(100), n_iters, replace=False) # for reproducibility
    acc_itr = []
    mae_itr = []
    std_itr = []
    selection_itr =[]
    reliab_itr =[]
    for iter in tqdm(range(n_iters)):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=seeds[iter], stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=2/7, random_state=seeds[iter], stratify=y_train)
        auc_grid = np.zeros((len(dropout_prob), len(f_correct)))
        loss_grid = np.zeros((len(dropout_prob), len(f_correct)))
        for i in range(len(dropout_prob)):
            for j in range(len(f_correct)):
                for r in range(n_repeats):
                    slc_g = GRACES(n_features=n_features, dropout_prob=dropout_prob[i], f_correct=f_correct[j])
                    selection_g = slc_g.select(x_train, y_train)
                    x_train_red_g = x_train[:, selection_g]
                    x_val_red = x_val[:, selection_g]
                    clf_g = svm.SVC(probability=True)
                    clf_g.fit(x_train_red_g, y_train)
                    y_val_pred = clf_g.predict_proba(x_val_red)
                    max_column_index = np.argmax(y_val_pred, axis=1) + 6
                    mae = np.absolute(y_val - max_column_index)
                    std_mae = np.std(mae)
                    auc_grid[i,j] += np.mean(mae)+std_mae
                    loss = torch.nn.CrossEntropyLoss()
                    loss_grid[i, j] += loss(torch.tensor(y_val_pred),torch.tensor(y_val).long()-6)
        index_i, index_j = np.where(auc_grid == np.max(auc_grid))
        best_index = np.argmin(loss_grid[index_i, index_j])
        best_prob, best_f_correct = dropout_prob[int(index_i[best_index])], f_correct[int(index_j[best_index])]
        acc = 0
        for r in range(n_repeats):
            slc = GRACES(n_features=n_features, dropout_prob=best_prob, f_correct=best_f_correct)
            selection = slc.select(x_train, y_train)
            x_train_red = x_train[:, selection]
            x_test_red = x_test[:, selection]
            clf = svm.SVC(probability=True)
            clf.fit(x_train_red, y_train)
            y_test_pred = clf.predict_proba(x_test_red)
            max_column_index = np.argmax(y_test_pred, axis=1) + 6
            mae = np.absolute(y_test - max_column_index)
            std_mae = np.std(mae)
            acc += std_mae + np.mean(mae)
        acc_itr.append(acc/n_repeats)
        mae_itr.append(mae)
        std_itr.append(std_mae)
        selection_itr.append(selection)
        reliab_itr.append(1-f_correct[index_j.item()])
    min_index = acc_itr.index(min(acc_itr))
    best_mae = mae_itr[min_index]
    best_std = std_itr[min_index]
    best_selection = selection_itr[min_index]
    best_reb = reliab_itr[min_index]
    return {'MAE': best_mae, 'STD': best_std, 'selction': best_selection, 'reliability': best_reb}


if __name__ == "__main__":
    name = 'Div2_MinMax_N564'
    max_features = 20
    n_iters = 2
    n_repeats = 2
    save = True

    for p in range(max_features):
        result = main(name=name, n_features=p+1, n_iters=n_iters, n_repeats=n_repeats)
        print('=========================== '+str(p+1)+' : Features'+' ==============================')
        print(' STD                                          : ' + str(result['STD']))
        print(' Reliability of GNN                           : ' + str(result['reliability']*100) + '%')
        print(' All indexs from Feature selections process : ' + str(result['selction']))
        print('========================================================================')
        if save:
            os.chdir(r'/home/ARO.local/netanelk/Projects/Featuer_selection_res/Normal_by_564_classif/')
            joblib.dump(result,str(p+1)+'_res.joblib')
            os.chdir(r'/home/ARO.local/netanelk/Projects/feature_selection_clas/')

