classdef person
    
    properties
        name;
        age;
    end
    
    methods
        function obj = person()
           obj.name = '';
           obj.age = 0;
        end
        
        function obj = change_name(obj,n)
            obj.name = n;
        end
        
        function obj = change_age(obj,a)
            obj.age = a;
        end
    end
end