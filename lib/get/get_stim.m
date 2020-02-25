function stim = get_stim(session, channel, unit, fold)
% Get stimuli of specific neuron
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
%
% Returns
% -------
% - stim: 2D matrix
%   (trial x time) response

result = load_result(session, channel, unit, fold);
stim = result.stim;
end