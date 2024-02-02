import catboost
from catboost import CatBoostClassifier
import pandas as pd
from scipy.stats import zscore
import pickle
import numpy as np

data = pd.read_csv('../data/data_0x22h_rankedImportanceRf.csv')
data_normalized = data.drop(columns=['dataSplit', 'label'])
data_normalized = data_normalized.apply(zscore)
data_normalized = data_normalized.iloc[:, 0:5]

data_best_model = pickle.load(open('../results/results_catboost_5features_strains.dat', 'rb'))
acc_val_array = data_best_model[0]
array_params = data_best_model[1]

max_post = np.where(acc_val_array == np.max(acc_val_array))[0][0]
param_best = array_params[max_post]
param_best['verbose'] = False

indices = pd.read_csv('indices_multi.csv')

model = []
accuracy_training = np.zeros(1000)
accuracy_validation = np.zeros(1000)
accuracy_test = np.zeros(1000)

sensitivity_training = np.zeros(1000)
sensitivity_validation = np.zeros(1000)
sensitivity_test = np.zeros(1000)

specificity_training = np.zeros(1000)
specificity_validation = np.zeros(1000)
specificity_test = np.zeros(1000)

for i in range(0, 1000):
    indices_current = indices.iloc[:, i]
    
    data_training = data_normalized[indices_current == 'training']
    label_training = data.label[indices_current == 'training']

    data_validation = data_normalized[indices_current == 'validation']
    label_validation = data.label[indices_current == 'validation']

    data_test = data_normalized[indices_current == 'test']
    label_test = data.label[indices_current == 'test']
    
    model.append(CatBoostClassifier(**param_best))

    model[i].fit(data_training, label_training, 
              eval_set = (data_validation, label_validation), 
              use_best_model = True);
    
    pred_training = model[i].predict(data_training)
    accuracy_training[i] = sum(pred_training == label_training) / float(len(label_training))
    sensitivity_training[i] = (sum((pred_training == label_training) & (label_training == 'Resistant')) / 
                           float(sum(label_training == 'Resistant')))
    specificity_training[i] = (sum((pred_training == label_training) & (label_training == 'Susceptible')) /
                           float(sum(label_training == 'Susceptible')))

    pred_validation = model[i].predict(data_validation)
    accuracy_validation[i] = sum(pred_validation == label_validation) / float(len(label_validation))
    sensitivity_validation[i] = (sum((pred_validation == label_validation) & (label_validation == 'Resistant')) / 
                              float(sum(label_validation == 'Resistant')))
    specificity_validation[i] = (sum((pred_validation == label_validation) & (label_validation == 'Susceptible')) / 
                              float(sum(label_validation == 'Susceptible')))

    pred_test = model[i].predict(data_test)
    accuracy_test[i] = sum(pred_test == label_test) / float(len(label_test))
    sensitivity_test[i] = (sum((pred_test == label_test) & (label_test == 'Resistant')) / 
                       float(sum(label_test == 'Resistant')))
    specificity_test[i] = (sum((pred_test == label_test) & (label_test == 'Susceptible')) /
                       float(sum(label_test == 'Susceptible')))
    print(i)	

data_to_save = [param_best, model, accuracy_training, accuracy_validation, accuracy_test, 
               sensitivity_training, sensitivity_validation, sensitivity_test, 
               specificity_training, specificity_validation, specificity_test]

pickle.dump(data_to_save, open('../results/results_catboost_5features_strains_multi.dat', 'wb'))

print('FNISHED')
