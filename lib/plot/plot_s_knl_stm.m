function plot_s_knl_stm(session, channel, unit, fold, prb)
% Plot stimulus kernel of specific s-model for given probe
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
% - prb: 2-by-1 integer vector
%   [x, y] postion of a probe

if nargin < 5
    plot_s_knl_stm_all_prbs(session, channel, unit, fold);
    return
end

stm = get_s_knls(session, channel, unit, fold);

% time-delay map
map = squeeze(stm(prb(1), prb(2), :, :));

create_figure('S-Model - Stimulus Kernel');

plot_time_delay_map(map);

title(sprintf('Stimulus kernel of S-Model for probe (%d, %d) and fold #%d', prb(1), prb(2), fold));

save_figure(...
    session, channel, unit, ...
    sprintf('s_knl_stm_prb_%d_%d_fold%02d', prb(1), prb(2), fold));
end