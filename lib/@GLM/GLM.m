classdef GLM < handle
    % Generalized Linear Model
    
    properties
        Coeff % Coefficients
    end
    
    % Constructor
    methods
        function this = GLM(coeff)
            this.Coeff = coeff;
        end
        
%         function y = predict(this, x)
%             b = this.Coeff;
%             y = 1 ./ (1 + exp(-(b(1) + b(2) .* x)));
%         end

        function y = predict(this, x)
            b = this.Coeff';
            N = size(x, 1); %? number of trials
            y = zeros(N, 1);
            for i = 1:N
                y(i) = 1 ./ (1 + exp(-dot(b, [1, x(i, :)])));
            end
        end
    end 
end