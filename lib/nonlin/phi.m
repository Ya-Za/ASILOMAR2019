function [y] = phi(x, param)
% scaled/biased logarithmic sigmoid: a * s(x - b)
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

y = param(1) .* logsig(x - param(2));
end

