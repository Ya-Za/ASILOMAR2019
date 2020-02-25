function [y] = dphi(x, param)
% Derivative of scaled/biased logarithmic sigmoid: a * s(x - b) (1 - s(x - b))
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

s = logsig(x - param(2));
y = param(1) .* s .* (1 - s);
end

