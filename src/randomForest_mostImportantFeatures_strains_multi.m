clear all
close all

load ../data/data_ranked_normalized.mat
load ../data/indices_strains_multi.mat

data_normalized = data_normalized(:, 1:5);
parfor i = 1:1000

data_training = data_normalized(indices_training{i}, :);
label_training =  label(indices_training{i}, :);

data_validation = data_normalized(indices_validation{i}, :);
label_validation = label(indices_validation{i});

data_test = data_normalized(indices_test{i}, :);
label_test = label(indices_test{i});

rf_model = TreeBagger(1000, data_training, label_training, ...
            'Method', 'classification', ...
            'MinLeafSize', 1, ...
            'MaxNumSplits', 1, ...
            'NumPredictorsToSample', 5);
        
prediction_training = predict(rf_model, data_training);
accuracy_training(i) = sum(label_training == prediction_training) / length(label_training);
sensitivity_training(i) = sum(label_training == prediction_training & label_training == 'Resistant') ...
    / sum(label_training == 'Resistant');
specificity_training(i) = sum(label_training == prediction_training & label_training == 'Susceptible') ...
    / sum(label_training == 'Susceptible');

prediction_validation = predict(rf_model, data_validation);
accuracy_validation(i) = sum(label_validation == prediction_validation) / length(label_validation);
sensitivity_validation(i) = sum(label_validation == prediction_validation & label_validation == 'Resistant') ...
    / sum(label_validation == 'Resistant');
specificity_validation(i) = sum(label_validation == prediction_validation & label_validation == 'Susceptible') ...
    / sum(label_validation == 'Susceptible');

prediction_test = predict(rf_model, data_test);
accuracy_test(i) = sum(label_test == prediction_test) / length(label_test);
sensitivity_test(i) = sum(label_test == prediction_test & label_test == 'Resistant') ...
    / sum(label_test == 'Resistant');
specificity_test(i) = sum(label_test == prediction_test & label_test == 'Susceptible') ...
    / sum(label_test == 'Susceptible');

[~, tmp] = predict(rf_model, data_normalized);

rf_prob(:, i) = tmp(:, 1);

fprintf('%i\n', i)
end

mean_acc_train = mean(accuracy_training);
mean_acc_val = mean(accuracy_validation);
mean_acc_test = mean(accuracy_test);

mean_sen_train = mean(sensitivity_training);
mean_sen_val = mean(sensitivity_validation);
mean_sen_test = mean(sensitivity_test);

mean_spe_train = mean(specificity_training);
mean_spe_val = mean(specificity_validation);
mean_spe_test = mean(specificity_test);

clear i data_test data_validation data_training prediction_test prediction_training prediction_validation tmp rf_model

save ../results/result_randomForest_5features_strains_multi.mat