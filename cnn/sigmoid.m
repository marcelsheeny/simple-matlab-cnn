classdef sigmoid < handle
    %SIGMOID Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_size;
        out_size;
    end
    
    methods
        function obj = sigmoid(in_size)
            obj.in_size = in_size;
            obj.out_size = in_size;
        end
        
        function data_f = forward(obj, data)
            data_f = sigmf(data,[1 0]);
        end

    end
    
end

