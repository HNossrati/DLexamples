classdef AutoDiff < handle
    properties
        id              % Unique string identifier
        value           % Numeric value of the tensor
        grad            % Gradient of the tensor
        requires_grad   % Whether gradient computation is needed
        parents         % Parent nodes in the computational graph
        backward_fn     % Function to compute local gradients
    end

    methods
        % Constructor
        function obj = AutoDiff(val, requires_grad)
            % Persistent counter for unique identifiers
            persistent counter
            if isempty(counter)
                counter = 0;
            end
            counter = counter + 1;
            obj.id = sprintf('node_%d', counter);

            % Default to not requiring gradient
            if nargin < 2
                requires_grad = false;
            end

            % Initialize properties
            obj.value = val;
            obj.grad = zeros(size(val), 'like', val);
            obj.requires_grad = requires_grad;
            obj.parents = {};
            obj.backward_fn = [];
        end

        % Backward pass method
        function backward(obj)
            % Initialize gradient to 1 if not set
            if all(obj.grad == 0)
                obj.grad = ones(size(obj.value), 'like', obj.value);
            end

            % Topological sort of computational graph
            sorted_nodes = obj.topological_sort();

            % Backward pass through sorted nodes
            for i = length(sorted_nodes):-1:1
                node = sorted_nodes{i};
                if ~isempty(node.backward_fn) && node.requires_grad
                    node.backward_fn();
                end
            end
        end

        % Topological sort for computational graph
        function sorted_nodes = topological_sort(obj)
            sorted_nodes = {};
            visited = containers.Map();

            function traverse(node)
                % Get unique hash for node
                node_hash = node.id;

                % Skip if already visited
                if isKey(visited, node_hash)
                    return;
                end

                % Mark as visited
                visited(node_hash) = true;

                % Recursively visit parents
                for j = 1:length(node.parents)
                    parent = node.parents{j};
                    if parent.requires_grad
                        traverse(parent);
                    end
                end

                % Add to sorted nodes
                sorted_nodes{end+1} = node;
            end

            % Start traversal from current node
            traverse(obj);
        end

        % Compute Mean Squared Error Loss

    end

    methods (Static)
        % Multiplication operation
        function c = mtimes(a, b)
            % Ensure inputs are AutoDiff objects
            if ~isa(a, 'AutoDiff')
                a = AutoDiff(a, false);
            end
            if ~isa(b, 'AutoDiff')
                b = AutoDiff(b, false);
            end

            % Compute value
            c_val = a.value * b.value;
            c = AutoDiff(c_val, a.requires_grad || b.requires_grad);

            % Add parents
            if a.requires_grad
                c.parents{end+1} = a;
            end
            if b.requires_grad
                c.parents{end+1} = b;
            end

            % Set backward function
            c.backward_fn = @() local_backward;

            function local_backward()
                % Gradient computation with accumulation
                if a.requires_grad
                    a.grad = a.grad + c.grad * b.value';
                end
                if b.requires_grad
                    b.grad = b.grad + a.value' * c.grad;
                end
            end
        end

        % Sigmoid activation
        function c = sigmoid(a)
            % Ensure input is AutoDiff object
            if ~isa(a, 'AutoDiff')
                a = AutoDiff(a, false);
            end

            % Compute sigmoid
            sig_val = 1 ./ (1 + exp(-a.value));
            c = AutoDiff(sig_val, a.requires_grad);

            % Add parent
            if a.requires_grad
                c.parents{end+1} = a;
            end

            % Set backward function
            c.backward_fn = @() local_backward;

            function local_backward()
                % Gradient computation for sigmoid
                if a.requires_grad
                    a.grad = a.grad + sig_val .* (1 - sig_val) .* c.grad;
                end
            end
        end

        % Subtraction operation
        function c = minus(a, b)
            % Ensure inputs are AutoDiff objects
            if ~isa(a, 'AutoDiff')
                a = AutoDiff(a, false);
            end
            if ~isa(b, 'AutoDiff')
                b = AutoDiff(b, false);
            end

            % Compute value
            c_val = a.value - b.value;
            c = AutoDiff(c_val, a.requires_grad || b.requires_grad);

            % Add parents
            if a.requires_grad
                c.parents{end+1} = a;
            end
            if b.requires_grad
                c.parents{end+1} = b;
            end

            % Set backward function
            c.backward_fn = @() local_backward;

            function local_backward()
                % Gradient computation
                if a.requires_grad
                    a.grad = a.grad + c.grad;
                end
                if b.requires_grad
                    b.grad = b.grad - c.grad;
                end
            end
        end

        % Power operation
        function c = power(a, n)
            % Ensure input is AutoDiff object
            if ~isa(a, 'AutoDiff')
                a = AutoDiff(a, false);
            end

            % Compute value
            c_val = a.value .^ n;
            c = AutoDiff(c_val, a.requires_grad);

            % Add parent
            if a.requires_grad
                c.parents{end+1} = a;
            end

            % Set backward function
            c.backward_fn = @() local_backward;

            function local_backward()
                % Gradient computation for power
                if a.requires_grad
                    a.grad = a.grad + n * (a.value .^ (n-1)) .* c.grad;
                end
            end
        end
        function loss = mse_loss(obj, targets)
            % Ensure inputs are AutoDiff objects
            if ~isa(obj, 'AutoDiff')
                obj = AutoDiff(obj, true);
            end
            if ~isa(targets, 'AutoDiff')
                targets = AutoDiff(targets, false);
            end

            % Compute MSE loss with explicit operations
            diff = AutoDiff.minus(obj, targets);
            squared = AutoDiff.power(diff, 2);
            
            % Take mean of squared errors
            loss = AutoDiff(mean(squared.value), true);
            loss.parents{1} = squared;
            loss.backward_fn = @() local_backward(squared, loss);

            function local_backward(squared_node, loss_node)
                % Backpropagate gradient through mean operation
                squared_node.grad = ones(size(squared_node.value), 'like', squared_node.value) ./ numel(squared_node.value);
            end
        end
    end
end