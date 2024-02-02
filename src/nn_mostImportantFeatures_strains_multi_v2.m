clear all
close all

load ../data/data_ranked_normalized.mat
load ../data/indices_strains_multi.mat

data_normalized = data_normalized(:, 1:5);
target = zeros(2, length(label));
target(1, label == 'Susceptible') = 1;
target(2, label == 'Resistant') = 1;

for i = 1:1000
    data_training = data_normalized(indices_training{i}, :);
    label_training =  label(indices_training{i}, :);

    data_validation = data_normalized(indices_validation{i}, :);
    label_validation = label(indices_validation{i});

    data_test = data_normalized(indices_test{i}, :);
    label_test = label(indices_test{i});
    
    net_tmp = cell(1, 100);
    acc_val_tmp = zeros(1, 100);
    acc_test_tmp = zeros(1, 100);
    per_tmp = zeros(1, 100);
    parfor j = 1:100
        net_tmp{j} = patternnet([18, 2]);
        net_tmp{j}.divideFcn = 'divideind';
        net_tmp{j}.divideParam.trainInd = indices_training{i};
        net_tmp{j}.divideParam.valInd = indices_validation{i};
        net_tmp{j}.divideParam.testInd = indices_test{i};

        net_tmp{j} = train(net_tmp{j}, ...
            data_normalized', target);

        prediction_validation = predict_net(net_tmp{j}, ...
            data_validation');
        acc_val_tmp(j) = sum(prediction_validation == label_validation) ...
            / length(label_validation);
        
        prediction_test = predict_net(net_tmp{j}, ...
            data_test');
        acc_test_tmp(j) = sum(prediction_test == label_test) ...
            / length(label_test);
        
        per_tmp(j) = (acc_val_tmp(j) + acc_test_tmp(j)) / 2;
    end
    
    X = find(per_tmp == max(per_tmp));
    X = X(1);

    net{i} = net_tmp{X};

    prediction_training = predict_net(net{i}, data_training');
    accuracy_training(i) = sum(label_training == prediction_training) / length(label_training);
    sensitivity_training(i) = sum(label_training == prediction_training & label_training == 'Resistant') ...
        / sum(label_training == 'Resistant');
    specificity_training(i) = sum(label_training == prediction_training & label_training == 'Susceptible') ...
        / sum(label_training == 'Susceptible');

    prediction_validation = predict_net(net{i}, data_validation');
    accuracy_validation(i) = sum(label_validation == prediction_validation) / length(label_validation);
    sensitivity_validation(i) = sum(label_validation == prediction_validation & label_validation == 'Resistant') ...
        / sum(label_validation == 'Resistant');
    specificity_validation(i) = sum(label_validation == prediction_validation & label_validation == 'Susceptible') ...
        / sum(label_validation == 'Susceptible');

    prediction_test = predict_net(net{i}, data_test');
    accuracy_test(i) = sum(label_test == prediction_test) / length(label_test);
    sensitivity_test(i) = sum(label_test == prediction_test & label_test == 'Resistant') ...
        / sum(label_test == 'Resistant');
    specificity_test(i) = sum(label_test == prediction_test & label_test == 'Susceptible') ...
        / sum(label_test == 'Susceptible');

    fprintf('%i\n', i)
end

clear j acc_val_tmp X net_tmp

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

save ../results/result_nn_5features_strains_multi_v2.mat