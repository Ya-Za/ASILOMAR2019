function plot_f_knl_stm(session, channel, unit, fold, prb)
% Plot stimulus kernel of specific f-model for given probe
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
% - prb: 2-by-1 integer vector
%   [x, y] postion of a probe

knl = get_f_knl_stm(session, channel, unit, fold);

% time-delay map
map = squeeze(knl(prb(1), prb(2), :, :));

create_figure('F-Model - Estimated Stimulus Kernel');

plot_time_delay_map(map);

title(sprintf('Estimated stimulus kernel with F-Model for probe (%d, %d) and fold #%d', prb(1), prb(2), fold));

save_figure(...
    session, channel, unit, ...
    sprintf('f_knl_stm_prb_%d_%d_fold%02d', prb(1), prb(2), fold));
end