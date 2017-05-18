classdef network < handle
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
    end
    
end

