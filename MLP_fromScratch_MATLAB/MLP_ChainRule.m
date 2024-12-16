clc; clear; close all
rng(1);


%% Generate training data
N_samples = 100; % Number of training samples (set of inputs)
N_featuresIN = 3;
N_featuresOut = 2;

X = randi([100 999], N_samples, N_featuresIN); % Training inputs
Y = zeros(N_samples, N_featuresOut); % True target

Y(:, 1) = sqrt(X(:, 1)); % waste
Y(:, 2) = X(:, 3) .* (X(:, 1) - Y(:, 1)) - X(:, 1) .* X(:, 2); % profit

%% Set hyperparameters
batch_size = 1; % Set batch size
epoch = 800;
learning_rate = 0.25;
Nodes = [N_featuresIN, 5, 4, N_featuresOut];

% Activation and derivative functions
% activation = @(x, alpha) tanh_func(x, alpha);
% d_activation = @(x, alpha) d_tanh(x, alpha);

activation = @(x) sigmoid_func(x, 1.5);
d_activation = @(x) d_sigmoid(x, 1.5);




%% Preprocessing
% Normalize data
normalization_mode = 'minmax'; %minmax, max, std
[X_norm, norm_params_X] = normalize_data(X, normalization_mode);
[Y_norm, norm_params_Y] = normalize_data(Y, normalization_mode);

% Split data
perm = randperm(N_samples);
X_norm_perm = X_norm(perm, :);
Y_norm_perm = Y_norm(perm, :);

TrainDataRatio = 0.8;

X_norm = X_norm_perm(1:floor(N_samples * TrainDataRatio), :); % Train set
Y_norm = Y_norm_perm(1:floor(N_samples * TrainDataRatio), :); % Train set

X_norm_valid = X_norm_perm((floor(N_samples * TrainDataRatio) + 1):end, :); % Validation set
Y_norm_valid = Y_norm_perm((floor(N_samples * TrainDataRatio) + 1):end, :); % Validation set

%% Training loop


% Initialize weights for each layer
weights = cell(length(Nodes) - 1, 1);
for i = 1:length(Nodes) - 1
    weights{i} = rand(Nodes(i), Nodes(i + 1)) - 0.5;
end

Losses_train = zeros(epoch, 1);
Losses_valid = zeros(epoch, 1);
num_batches = ceil(length(X_norm) / batch_size);

for e = 1:epoch
    % Shuffle the data at the beginning of each epoch
    perm = randperm(length(X_norm));
    X_norm_perm = X_norm(perm, :);
    Y_norm_perm = Y_norm(perm, :);

    for b = 1:num_batches
        % Get minibatch data
        batch_start = (b - 1) * batch_size + 1;
        batch_end = min(b * batch_size, N_samples);
        X_batch = X_norm_perm(batch_start:batch_end, :);
        Y_batch = Y_norm_perm(batch_start:batch_end, :);

        % Forward pass
        [Y_prd, A] = forward_func(X_batch, weights, activation);

        % Backpropagation
        d_weights = backpropagation(Y_batch, weights, A, d_activation);

        % Gradient descent update
        for i = 1:length(weights)
            
            weights{i} = weights{i} - learning_rate * (d_weights{i});
        end
    end

    % Compute loss for the entire train dataset
    [Y_prd_train, ~] = forward_func(X_norm, weights, activation);
    L_train = sum(sum((Y_prd_train - Y_norm).^2)) / numel(Y_prd_train);
    Losses_train(e) = L_train;

    % Compute loss for the validation dataset
    [Y_prd_valid, ~] = forward_func(X_norm_valid, weights, activation);
    L_valid = sum(sum((Y_prd_valid - Y_norm_valid).^2)) / numel(Y_norm_valid);
    Losses_valid(e) = L_valid;

    if mod(e, 50) == 0
        fprintf('Epoch: %d Train Loss: %.4f Validation Loss: %.4f\n', ...
                e, L_train, L_valid);
    end
end


%% Evaluate model

% Plot Loss
figure; hold on
    plot(log(Losses_train))
    plot(log(Losses_valid))
    title('Loss vs Iteration')
    legend('Train', 'Validation')

% Plot Target vs Prediction for each output
figure; hold on
    scatter(Y_norm(:, 1), Y_prd_train(:, 1));
    plot(linspace(min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1)), N_samples), ...
        linspace(min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1)), N_samples))
    title('Target vs. Prediction 1')
    xlim([min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1))])
    ylim(([min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1))]))

figure; hold on
    scatter(Y_norm(:, 2), Y_prd_train(:, 2));
    plot(linspace(min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2)), N_samples), ...
        linspace(min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2)), N_samples))
    title('Target vs. Prediction 2')
    xlim([min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2))])
    ylim(([min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2))]))

figure; hold on
    scatter(1:length(X_norm), Y_prd_train(:,1), 'ok');
    scatter(1:length(X_norm), Y_norm(:,1), '.r'),
    title('Target and Prediction 1')
    legend('Prediction', 'Target')

figure; hold on
    scatter(1:length(X_norm), Y_prd_train(:, 2), 'ok');
    scatter(1:length(X_norm), Y_norm(:, 2), '.r'),
    title('Target and Prediction 2')
    legend('Prediction', 'Target')


% Test case
X_test = [444 555 777];
Y_test = zeros(2, 1);

Y_test(1) = sqrt(X_test(1));
Y_test(2) = (X_test(1) - Y_test(1)) * X_test(3) - X_test(2) * X_test(1);

% Normalize test data
X_test_norm = normalize_data(X_test, normalization_mode, norm_params_X);

% Forward pass for test
[Y_test_pred_norm, ~] = forward_func(X_test_norm, weights, activation);

% Denormalize predictions
Y_test_pred = denormalize_data(Y_test_pred_norm, normalization_mode, norm_params_Y);

fprintf('Target   output 1 = %6.1f \n', Y_test(1)); fprintf('\n')
fprintf('Target   output 2 = %6.1f \n', Y_test(2)); fprintf('\n')
fprintf('Predicted output 1 = %6.1f \n', Y_test_pred(1)); fprintf('\n')
fprintf('Predicted output 2 = %6.1f \n', Y_test_pred(2)); fprintf('\n')
fprintf('Test error 1  = %0.5f \n', abs(Y_test(1) - Y_test_pred(1))); fprintf('\n')
fprintf('Test error 2  = %0.5f \n', abs(Y_test(2) - Y_test_pred(2)));


%% Helper Functions

% Activation Functions
function x = sigmoid_func(x, alpha)
    x = 1 ./ (1 + exp(-alpha * x)); % sigmoid
end

function x = d_sigmoid(x, alpha)
    x = alpha * x .* (1 - x); % Derivative of the sigmoid function
end

function x = tanh_func(x, alpha)
    x = (1 - exp(-alpha * x)) ./ (1 + exp(-alpha * x)); % tanh
end

function x = d_tanh(x, alpha)
    x = alpha * (1 - x.^2); % Derivative of tanh
end


% Forward Function
function [Y, A] = forward_func(X, weights, activation)
    % Feed Forward
    A = cell(length(weights) + 1, 1);
    A{1} = X;

    for i = 1:(length(weights))
        M = A{i} * weights{i};
        if i == (length(weights))
            A{i + 1} = M;
        else
            A{i + 1} = activation(M);
        end
    end

    Y = A{end};
end

% Backward Function
function  d_weights = backpropagation(Y, weights, A, d_activation)
    % Backpropagation and weight update
    delta = cell(length(weights), 1);

    % Output layer error
    delta{end} = (A{end} - Y);

    % Hidden layer errors
    for i = (length(weights) - 1):-1:1
        error = delta{i + 1} * weights{i + 1}';
        delta{i} = error .* d_activation(A{i + 1});
    end
    
    d_weights = cell(length(weights), 1);

    for i = 1:length(weights)
        d_weights{i} = A{i}' * delta{i};
    end

end


function [data_norm, norm_params] = normalize_data(data, method, norm_params)
    % Normalization function
    if nargin < 3
        switch method
            case 'max'
                norm_params.max = max(data, [], 1);
                data_norm = data ./ norm_params.max;
            case 'minmax'
                norm_params.min = min(data, [], 1);
                norm_params.max = max(data, [], 1);
                data_norm = (data - norm_params.min) ./ (norm_params.max - norm_params.min);
            case 'std'
                norm_params.mean = mean(data, 1);
                norm_params.std = std(data, [], 1);
                data_norm = (data - norm_params.mean) ./ norm_params.std;
        end
    else
        switch method
            case 'max'
                data_norm = data ./ norm_params.max;
            case 'minmax'
                data_norm = (data - norm_params.min) ./ (norm_params.max - norm_params.min);
            case 'std'
                data_norm = (data - norm_params.mean) ./ norm_params.std;
        end
    end
end

function data_denorm = denormalize_data(data_norm, method, norm_params)
    % Denormalization function
    switch method
        case 'max'
            data_denorm = data_norm .* norm_params.max;
        case 'minmax'
            data_denorm = data_norm .* (norm_params.max - norm_params.min) + norm_params.min;
        case 'std'
            data_denorm = data_norm .* norm_params.std + norm_params.mean;
    end
end
