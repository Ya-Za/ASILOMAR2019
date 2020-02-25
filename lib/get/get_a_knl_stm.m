function knl = get_a_knl_stm(session, channel, unit)
% Get average of estimated stimulus kernels of specific a-model
%
% Parameters
% ----------
%
%
% Returns
% -------
% - knl: 4D array
%   (width x height x time x delay) stimulus kernel of s-model

model = load_avg_model(session, channel, unit);
knl = model.set_of_kernels.gas.knl;
end