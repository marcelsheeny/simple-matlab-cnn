classdef dense < handle
    %RELU Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        out_size;
        w;
        grad;
        data_out;
    end
    
    methods
        function obj = dense(data_in_size, out_size)
           obj.out_size = out_size;
           scale = sqrt(1.0/prod(data_in_size(:)));
           obj.w = rand(data_in_size,out_size);
        end
        
        function data_f = forward(obj, data)
            obj.data_out = zeros(obj.out_size,1);
            for i=1:obj.out_size
                obj.data_out(i,1) = dot(obj.w(:,i), data(:));
            end
            data_f = obj.data_out;
        end
        
        function out = backward(obj, err, data_out)
            obj.grad = (data_out > 0);
            prop_err = obj.w *  err;
            out = prop_err .* obj.grad;
            obj.grad = out;
        end

    end
    
end



