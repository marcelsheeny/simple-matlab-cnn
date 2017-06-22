classdef relu < handle
    %RELU Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_size;
        out_size;
        grad;
    end
    
    methods
        function obj = relu(in_size)
            obj.in_size = in_size;
            obj.out_size = in_size;
        end
        
        function data_f = forward(obj, data)
            data_f = max(0, data);
        end
        
        function backward(obj, grad)
            obj.grad = grad > 0;
        end

    end
    
end

