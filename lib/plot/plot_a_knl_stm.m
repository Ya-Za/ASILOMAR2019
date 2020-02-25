function plot_a_knl_stm(session, channel, unit, prb)
% Plot average of stimulus kernels of specific a-model for given probe
%
% Parameters
% ----------
%
% - prb: 2-by-1 integer vector
%   [x, y] postion of a probe

knl = get_a_knl_stm(session, channel, unit);

% time-delay map
map = squeeze(knl(prb(1), prb(2), :, :));

create_figure('A-Model - Average of Estimated Stimulus Kernels');

plot_time_delay_map(map);

title(sprintf('Average of estimated stimulus kernels with A-Model for probe (%d, %d)', prb(1), prb(2)));

save_figure(...
    session, channel, unit, ...
    sprintf('a_knl_stm_prb_%d_%d_fold%02d', prb(1), prb(2), fold));
end