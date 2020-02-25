function y = logsig(x)
% Logarithmic sigmoid
%
% Parameters
% ----------
% - x: array
%   Input
%
% Returns
% -------
% - y: array
%   Output

y = 1 ./ (1 + exp(-x));
end
