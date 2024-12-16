% Neural Network with Custom Automatic Differentiation
% Clear previous variables and set random seed
clc; clear; close all;
rng(1);

%% Generate training data
N_samples = 100; % Number of training samples
N_featuresIN = 3;
N_featuresOut = 2;

% Generate input data
X_raw = randi([100 999], N_samples, N_featuresIN);

% Compute true targets
Y_raw = zeros(N_samples, N_featuresOut);
Y_raw(:, 1) = sqrt(X_raw(:, 1)); % waste
Y_raw(:, 2) = X_raw(:, 3) .* (X_raw(:, 1) - Y_raw(:, 1)) - X_raw(:, 1) .* X_raw(:, 2); % profit

%% Preprocessing
% Normalize data
[X, X_norm_params] = normalize_data(X_raw, 'minmax');
[Y, Y_norm_params] = normalize_data(Y_raw, 'minmax');

% Split data
TrainDataRatio = 0.8;
X_train = X(1:floor(N_samples * TrainDataRatio), :);
Y_train = Y(1:floor(N_samples * TrainDataRatio), :);
X_valid = X((floor(N_samples * TrainDataRatio) + 1):end, :);
Y_valid = Y((floor(N_samples * TrainDataRatio) + 1):end, :);

%% Network Hyperparameters
batch_size = 1;
epochs = 800;
learning_rate = 0.25;
layer_sizes = [N_featuresIN, 5, 4, N_featuresOut];

%% Initialize Weights with AutoDiff
weights = cell(length(layer_sizes) - 1, 1);
for i = 1:length(layer_sizes) - 1
    % Create weights as AutoDiff objects with gradient tracking
    weights{i} = AutoDiff(rand(layer_sizes(i), layer_sizes(i+1)) - 0.5, true);
    
end

%% Training Loop
train_losses = zeros(epochs, 1);
valid_losses = zeros(epochs, 1);

for epoch = 1:epochs
    % Shuffle training data at the start of each epoch
    shuffle_idx = randperm(size(X_train, 1));
    X_shuffled = X_train(shuffle_idx, :);
    Y_shuffled = Y_train(shuffle_idx, :);
    
    for b = 1:ceil(size(X_train, 1) / batch_size)
        % Get batch
        batch_start = (b-1) * batch_size + 1;
        batch_end = min(b * batch_size, size(X_train, 1));
        X_batch = X_shuffled(batch_start:batch_end, :);
        Y_batch = Y_shuffled(batch_start:batch_end, :);
        
        % Clear previous gradients
        for i = 1:length(weights)
            weights{i}.grad = zeros(size(weights{i}.value));
        end
        
        % Forward pass
        activations = forward_pass(X_batch, weights, true);
        
        % Compute loss
        loss = AutoDiff.mse_loss(activations{end}, Y_batch);
        
        % Backward pass
        loss.backward();
        
        % Update weights with gradient clipping
        for i = 1:length(weights)
            % Clip gradients to prevent explosion
            max_grad_norm = 1.0;
            grad_norm = norm(weights{i}.grad(:));
            if grad_norm > max_grad_norm
                weights{i}.grad = weights{i}.grad * (max_grad_norm / grad_norm);
            end
            
            % Update weights
            weights{i}.value = weights{i}.value - learning_rate * weights{i}.grad;

        end
        
        % Print batch loss occasionally
%         if mod(b, 10) == 0
%             fprintf('Epoch %2d, Batch %2d, Loss: %.4f\n', epoch, b, mean(loss.value));
% 
%         end
    end
    
    % Compute training loss
    train_pred = forward_pass(X_train, weights, false);
    
    train_losses(epoch) = mean(compute_metric(train_pred{end}, Y_train));
    
    % Compute validation loss
    valid_pred = forward_pass(X_valid, weights, false);
    valid_losses(epoch) = mean(compute_metric(valid_pred{end}, Y_valid));
    if mod(epoch, 50) == 0
        fprintf('Epoch: %d Train Loss: %.4f Validation Loss: %.4f\n', ...
                epoch, train_losses(epoch), valid_losses(epoch));
    end
end





%% Evaluate model
Y_prd_train = train_pred{end}.value;
Y_prd_valid = valid_pred{end}.value;
% Plot Loss
figure; hold on
    plot(log(train_losses))
    plot(log(valid_losses))
    title('Loss vs Iteration')
    legend('Train', 'Validation')

% Plot Target vs Prediction for each output
figure; hold on
    scatter(Y_train(:, 1), Y_prd_train(:, 1));
    plot(linspace(min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1)), N_samples), ...
        linspace(min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1)), N_samples))
    title('Target vs. Prediction 1')
    xlim([min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1))])
    ylim(([min(Y_prd_train(:, 1)), max(Y_prd_train(:, 1))]))

figure; hold on
    scatter(Y_train(:, 2), Y_prd_train(:, 2));
    plot(linspace(min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2)), N_samples), ...
        linspace(min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2)), N_samples))
    title('Target vs. Prediction 2')
    xlim([min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2))])
    ylim(([min(Y_prd_train(:, 2)), max(Y_prd_train(:, 2))]))

figure; hold on
    scatter(1:length(X_train), Y_prd_train(:,1), 'ok');
    scatter(1:length(X_train), Y_train(:,1), '.r'),
    title('Target and Prediction 1')
    legend('Prediction', 'Target')

figure; hold on
    scatter(1:length(X_train), Y_prd_train(:, 2), 'ok');
    scatter(1:length(X_train), Y_train(:, 2), '.r'),
    title('Target and Prediction 2')
    legend('Prediction', 'Target')


%% Test case
X_test = [444 555 777];
Y_test = zeros(2, 1);

Y_test(1) = sqrt(X_test(1));
Y_test(2) = (X_test(1) - Y_test(1)) * X_test(3) - X_test(2) * X_test(1);

% Normalize test data


X_test_norm = normalize_data(X_test, "minmax", X_norm_params);

% Forward pass for test
Y_test_pred_norm = forward_pass(X_test_norm,  weights, false);
Y_test_pred_norm = Y_test_pred_norm{end}.value;
% Denormalize predictions
Y_test_pred = denormalize_data(Y_test_pred_norm, "minmax", Y_norm_params);

fprintf('Target   output 1 = %6.1f \n', Y_test(1)); fprintf('\n')
fprintf('Target   output 2 = %6.1f \n', Y_test(2)); fprintf('\n')
fprintf('Predicted output 1 = %6.1f \n', Y_test_pred(1)); fprintf('\n')
fprintf('Predicted output 2 = %6.1f \n', Y_test_pred(2)); fprintf('\n')
fprintf('Test error 1  = %0.5f \n', abs(Y_test(1) - Y_test_pred(1))); fprintf('\n')
fprintf('Test error 2  = %0.5f \n', abs(Y_test(2) - Y_test_pred(2)));

%% Helper Functions

function activations = forward_pass(X_batch, weights, is_training)
    % Forward pass using enhanced AutoDiff
    activations = cell(length(weights) + 1, 1);
    
    % First layer activation is the input
    activations{1} = AutoDiff(X_batch, is_training);
    
    % Iterate through layers
    for i = 1:length(weights)
        % Ensure weights are AutoDiff objects with gradient tracking
        if ~isa(weights{i}, 'AutoDiff')
            weights{i} = AutoDiff(weights{i}, true);
        end
        
        % Matrix multiplication
        z = AutoDiff.mtimes(activations{i}, weights{i});
        
        % Apply sigmoid activation for hidden layers
        if i < length(weights)
            activations{i+1} = AutoDiff.sigmoid(z);
        else
            % Linear output for the last layer
            activations{i+1} = z;
        end
    end
end


function loss = compute_metric(predictions, targets)
    % Standard MATLAB loss computation for evaluation
    if isa(predictions, 'AutoDiff')
        predictions = predictions.value;
    end
    diff = predictions - targets;
    loss = mean(diff.^2);

end


% Normalization functions (same as in original script)

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
