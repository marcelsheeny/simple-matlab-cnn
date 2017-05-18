classdef input_layer < handle
    %INPUT_DATA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_size;
        out_size;
    end
    
    methods
        function obj = input_layer(in_size)
           obj.in_size = in_size;
           obj.out_size = in_size;
        end
        
        function out = forward(obj, data)
            out = data;
        end
    end
    
end

