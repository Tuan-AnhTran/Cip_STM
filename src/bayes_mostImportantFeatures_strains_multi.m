clear all
close all

load ../data/data_ranked_normalized.mat
load ../data/indices_strains_multi.mat

data_normalized = data_normalized(:, 1:10);

for i = 1:1000
    data_training = data_normalized(indices_training{i}, :);
    label_training =  label(indices_training{i}, :);

    data_validation = data_normalized(indices_validation{i}, :);
    label_validation = label(indices_validation{i});

    data_test = data_normalized(indices_test{i}, :);
    label_test = label(indices_test{i});

    bayes_model{i} = fitcnb(data_training, label_training);

    prediction_training = predict(bayes_model{i}, data_training);
    accuracy_training(i) = sum(label_training == prediction_training) / length(label_training);
    sensitivity_training(i) = sum(label_training == prediction_training & label_training == 'Resistant') ...
        / sum(label_training == 'Resistant');
    specificity_training(i) = sum(label_training == prediction_training & label_training == 'Susceptible') ...
        / sum(label_training == 'Susceptible');

    prediction_validation = predict(bayes_model{i}, data_validation);
    accuracy_validation(i) = sum(label_validation == prediction_validation) / length(label_validation);
    sensitivity_validation(i) = sum(label_validation == prediction_validation & label_validation == 'Resistant') ...
        / sum(label_validation == 'Resistant');
    specificity_validation(i) = sum(label_validation == prediction_validation & label_validation == 'Susceptible') ...
        / sum(label_validation == 'Susceptible');


    prediction_test = predict(bayes_model{i}, data_test);
    accuracy_test(i) = sum(label_test == prediction_test) / length(label_test);
    sensitivity_test(i) = sum(label_test == prediction_test & label_test == 'Resistant') ...
        / sum(label_test == 'Resistant');
    specificity_test(i) = sum(label_test == prediction_test & label_test == 'Susceptible') ...
        / sum(label_test == 'Susceptible');

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

clear i data_test data_validation data_training prediction_test prediction_training prediction_validation

save ../results/result_bayes_10features_strains_multi.mat