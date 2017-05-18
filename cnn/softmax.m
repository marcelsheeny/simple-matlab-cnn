classdef softmax < handle
    %RELU Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_size;
        out_size;
    end
    
    methods
        function obj = softmax(in_size)
            obj.in_size = in_size;
            obj.out_size = in_size;
        end
        
        function data_f = forward(obj, data)
            amax = max(data(:));
            es = exp(data(:) - amax);
            esum = sum(es);
            data_f = es/esum;
        end

    end
    
end

