classdef conv_layer < handle
    %NETWORK Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        w;            % convolution weights
        b;            % convolution bias
        in_size;      % input size
        out_size;     % output_size (i,j,k)
        filter_size;  % filter size (i,j)
        data_out;

    end
    
    methods
        function obj = conv_layer(filter_size, filters, in_size)
            obj.in_size = in_size;
            obj.filter_size = [filter_size, in_size(3), filters];
            obj.out_size = [in_size(1), in_size(2), filters];
            scale = sqrt(1.0/prod(obj.filter_size(:)));
            obj.w = scale*rand(obj.filter_size);
            obj.b = scale*rand(filters,1);

        end
        
        function data_f = forward(obj, data)
            obj.data_out = zeros(obj.out_size);
            for i=1:obj.out_size(3)
                obj.data_out(:,:,i) = convn(data, obj.w(:,:,:,i), 'same') + obj.b(i);
            end
            data_f = obj.data_out;
        end
    end
    
end

