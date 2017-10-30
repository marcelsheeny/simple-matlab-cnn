classdef relu < handle
    %RELU Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_size;
        out_size;
        grad;
        data_out;
    end
    
    methods
        function obj = relu(in_size)
            obj.in_size = in_size;
            obj.out_size = in_size;
        end
        
        function data_f = forward(obj, data)
            % obj.w = data;
            obj.data_out = max(0, data);
            data_f = obj.data_out;
        end
        
        function out = backward(obj, grad, data_out)
            out = grad;
            obj.grad = grad;
        end

    end
    
end

