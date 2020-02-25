function knl = get_f_knl_stm(session, channel, unit, fold)
% Get estimated stimulus kernel of specific f-model
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
%
% Returns
% -------
% - knl: 4D array
%   (width x height x time x delay) stimulus kernel of s-model

model = load_model(session, channel, unit, fold);
knl = model.set_of_kernels.gas.knl;
end