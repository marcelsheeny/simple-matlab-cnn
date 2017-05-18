classdef max_pool < handle
    %MAX_POOL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        in_size;
        filter_size;
        out_size;
    end
    
    methods
        function obj = max_pool(siz, in_size)
           obj.filter_size = siz;
           obj.out_size = [in_size(1)/2, in_size(2)/2, in_size(3)];
           obj.in_size = in_size;
        end
        
        function data_f = forward(obj, data)
            data_f = zeros(obj.out_size);
            for i=1:size(data_f,1)
                for j=1:size(data_f,2)
                    for k=1:size(data_f,3)
                        pacth = data(i*2-obj.filter_size(1)/2:i*2, ...
                                     j*2-obj.filter_size(2)/2:j*2,k);
                        data_f(i,j,k) = max(pacth(:));
                    end
                end
            end
        end

    end
    
end

