classdef neural_network < handle
    %NETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        layers;
    end
    
    methods
        function obj = network()
           obj.layers = {};
        end
        
        function obj = add_layer(obj, varargin)
            
            for i=1:length(varargin)
                if (strcmp(varargin{i}, 'type')) type = varargin{i+1}; end
                if (strcmp(varargin{i}, 'filters')) filters = varargin{i+1}; end
                if (strcmp(varargin{i}, 'size')) siz = varargin{i+1}; end
                if (strcmp(varargin{i}, 'drop_prob')) drop_prob = varargin{i+1}; end
            end
            
            if (~strcmp(type, 'input'))
                in_size = obj.layers{end}.out_size;
            end
            if (strcmp(type, 'conv')) layer = conv_layer(siz, filters, in_size); end
            if (strcmp(type, 'input')) layer = input_layer(siz); end
            if (strcmp(type, 'max_pool')) layer = max_pool(siz, in_size); end
            if (strcmp(type, 'relu')) layer = relu(in_size); end
            %if (strcmp(type, 'dropout')) layer = dropout(drop_prob); end
            if (strcmp(type, 'softmax')) layer = softmax(prod(in_size(:))); end
            if (strcmp(type, 'dense')) layer = dense(prod(in_size(:)), siz); end
            obj.layers{end + 1} = layer; 
        end
        
        function data = forward(obj, data)
            for i=1:length(obj.layers)
                data = obj.layers{i}.forward(data); 
            end
        end
        
        function backward(obj, err)
            last_layer = length(obj.layers);
            grad = obj.layers{last_layer}.backward(err);
            for i=last_layer-1:-1:1
                grad = obj.layers{i}.backward(grad); 
            end
        end
        
        function [all_err_train, all_err_val] = train(obj, epochs, l_r, batch, data, labels, val_percent, epoch_val)
           [trainInd,valInd,~] = dividerand(size(data,1),1 - val_percent,val_percent,0);
           all_err_train = [];
           all_err_val = [];
           train_data = data(trainInd,:,:);
           train_labels = labels(trainInd);
           val_data = data(valInd,:,:);
           val_labels = labels(valInd);
           curr_batch = 1;
           max_label = max(labels(:));
           for i=1:epochs
               if (i == mod(epochs, epoch_val))
                   err_val = 0;
                   for v=1:length(valInd)
                       f = obj.forward(val_data(:,:,b));
                       truth = zeros(max_label,1);
                       truth(val_labels(b)) = 1;
                       err_val = err_val + sum(abs(f - truth));
                   end
                   all_err_val(end+1,1) = err_val;
                   err_train = 0;
                   for v=1:length(trainInd)
                       f = obj.forward(train_data(:,:,b));
                       truth = zeros(max_label,1);
                       truth(train_labels(b)) = 1;
                       err_train = err_train + sum(abs(f - truth));
                   end
                   all_err_train(end+1,1) = err_val;
               else
                   err = 0;
                   for b=curr_batch:curr_batch+batch
                       if (b < size(data,1))
                           f = obj.forward(train_data(:,:,b));
                           truth = zeros(max_label,1);
                           truth(train_labels(b)) = 1;
                           err = err + sum(abs(f - truth));
                       end
                       curr_batch = b;
                       obj.backward(err);
                       
                       %update weights
                       last_layer = length(obj.layers);
                       for l=last_layer-1:-1:1
                           obj.layers{l}.w = obj.layers{l}.w + l_r*obj.layers{l}.grad;
                       end
                   end
               end
           end
        end
    end
    
end

