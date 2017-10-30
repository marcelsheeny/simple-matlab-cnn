classdef softmax < handle
    %RELU Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_size;
        out_size;
        curr_data;
        grad;
        data_out;
    end
    
    methods
        function obj = softmax(in_size)
            obj.in_size = in_size;
            obj.out_size = in_size;
        end
        
        function data_f = forward(obj, data)
            obj.curr_data = data(:);
            amax = max(obj.curr_data);
            esum = 0;
            es = zeros(length(obj.curr_data),1);
            for i=1:length(obj.curr_data)
                es(i) = exp(obj.curr_data(i) - amax);
                esum = esum + es(i);
            end
            obj.data_out = es/esum;
            data_f = obj.data_out;
            
        end
        
        function out = backward(obj, grad)
            obj.grad = grad;
            out = grad;
        end
        
    end
    
end

