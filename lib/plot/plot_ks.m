function plot_ks(session, channel, unit, fold, tlim)
% Kolmogorov Smirnov (KS) plot of specific s-model for given period
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
% - tlim: [scalar, scalar]
%   Time limits from saccade
% - probe: 2-by-1 integer vector
%   [x, y] postion of a probe

line_width = 3; % width of line

% response
resp = get_resp(session, channel, unit, fold);
T = size(resp, 2); % number of times
tsac = ceil(T / 2); % saccade time
times = (tlim(1):tlim(2)) + tsac; % time from saccade onset
resp = resp(:, times);

% s-model firing rate
s_fr = get_s_fr(session, channel, unit, fold);
s_fr = s_fr(:, times);
s_fr = s_fr / 1000;


[~, rstsort, xks, cb, ~] = ksdiscrete(s_fr(:), resp(:), 'spiketrain');

% plot
create_figure('KS');

% % y = x
% plot([0, 1], [0, 1], 'Color', [0, 0, 0]);

% bounds
% - upper bound
plot(xks, xks + cb, 'Color', [0, 0, 0]);
hold('on');
% - upper bound
plot(xks, xks - cb, 'Color', [0, 0, 0]);

% continous time
h1 = plot(xks, xks, ...
    'LineStyle', '--', ...
    'linewidth', line_width, ...
    'Color', [0.4, 0.4, 0.4], ...
    'Displayname', 'Continuous Time');

% discrete time
h2 = plot(xks, rstsort, ...
    'LineStyle', '-', ...
    'linewidth', line_width, ...
    'Color', [0.4, 0.4, 0.4], ...
    'Displayname', 'Discrete Time');

xticks([0, 0.5, 1]);
yticks([0, 0.5, 1]);

title(sprintf('Kolmogorov Smirnov (KS) Plot\nat period (%d, %d)', tlim(1), tlim(2)));
xlabel('empirical CDF(z)');
ylabel('uniform CDF');
% legend([h1, h2], {'Continuous Time', 'Discrete Time', }, 'Location', 'SE');

set_axis();
axis([0, 1, 0, 1]);

save_figure(...
    session, channel, unit, ...
    sprintf('ks_tlim_%d_%d_fold%02d', tlim(1), tlim(2), fold));
end
