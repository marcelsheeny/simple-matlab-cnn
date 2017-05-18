classdef dense < handle
    %RELU Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        out_size;
        w;
    end
    
    methods
        function obj = dense(data_in_size, out_size)
           obj.out_size = out_size;
           scale = sqrt(1.0/prod(data_in_size(:)));
           obj.w = rand(data_in_size,out_size);
        end
        
        function data_f = forward(obj, data)
            data_f = zeros(obj.out_size,1);
            for i=1:obj.out_size
                data_f(i,1) = dot(obj.w(:,i), data(:));
            end
        end

    end
    
end



