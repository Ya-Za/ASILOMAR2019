function plot_mean_resp(session, channel, unit, fold, tlim, prb, win)
% Plot response of specific s-model for given trial
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
% - tlim: [scalar, scalar]
%   Time limits from saccade
% - prb: 2-by-1 integer vector
%   [x, y] postion of a probe
% - window: scalar
%   Length of response window

sw = 21; % smoothing window
colors = lines(2); % color
line_width = 3; % width of line
alpha = 0.2; % color alpha

times = 1:win;

[mean_resp, se_resp, mean_s_fr, se_s_fr] = ...
    get_mean_resp(session, channel, unit, fold, tlim, prb, win);


mean_resp = smoothdata(mean_resp, 'gaussian', sw);
se_resp = smoothdata(se_resp, 'gaussian', sw);
mean_s_fr = smoothdata(mean_s_fr, 'gaussian', sw);
se_s_fr = smoothdata(se_s_fr, 'gaussian', sw);


create_figure('Mean Response');

% resp
plot(times, mean_resp, ...
    'DisplayName', '  E[Response]', ...
    'LineWidth', line_width, ...
    'Color', colors(1, :));
hold('on');
patch(...
    'XData', [times, flip(times)], ...
    'YData', [mean_resp - se_resp, flip(mean_resp + se_resp)], ...
    'DisplayName', 'SE[Response]', ...
    'LineStyle', 'none', ...
    'FaceColor', colors(1, :), ...
    'FaceAlpha', alpha);

% s-model firing rate
plot(mean_s_fr, ...
    'DisplayName', '  E[S-Model]', ...
    'LineWidth', line_width, ...
    'Color', colors(2, :));
patch(...
    'XData', [times, flip(times)], ...
    'YData', [mean_s_fr - se_s_fr, flip(mean_s_fr + se_s_fr)], ...
    'DisplayName', 'SE[S-Model]', ...
    'LineStyle', 'none', ...
    'FaceColor', colors(2, :), ...
    'FaceAlpha', alpha);
hold('off');

xticks([1, 50:50:win]);
yticks([0, 100]);

title(sprintf('Mean response for probe (%d, %d) at period (%d, %d)', ...
    prb(1), prb(2), tlim(1), tlim(2)));
xlabel('Time (ms)');
ylabel('Firing rate (spk/s)');
legend();

set_axis();

save_figure(...
    session, channel, unit, ...
    sprintf('mean_resp_prb_%d_%d_tlim_%d_%d_fold%02d', prb(1), prb(2), tlim(1), tlim(2), fold));
end
