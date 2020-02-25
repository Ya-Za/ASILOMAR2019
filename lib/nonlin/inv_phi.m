function [y] = inv_phi(x, param)
% Inverse of scaled/biased logarithmic sigmoid: b - log((a / x) - 1)
%
% Parameters
% ----------
% - x: array
%   Input
% - param: [scalar, scalar]
%   Scale and bias parameters
%
% Returns
% -------
% - y: array
%   Output

y = param(2) - log((param(1) ./ x) - 1);
end
