function plot_s_fr(session, channel, unit, fold, type_trials)
% Plot response firting rate of specific s-model for given trial
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
% - type_trials: char vector | scalar
%   Trials type or trial index

info = get_info();
times = info.times;

s = 5; % scale spikes
colors = lines(2);

s_fr = get_s_fr(session, channel, unit, fold);

if ischar(type_trials)
    trials = get_trials(session, channel, unit, fold, type_trials);
else
    trials = type_trials;
    type_trials = num2str(type_trials);
end

s_fr = s_fr(trials, :);

create_figure('S-Model - Response');

% firing rate
plot(times, mean(s_fr, 1), ...
    'LineWidth', 2, ...
    'Color', colors(1, :));

xticks([times(1), 0, times(end)]);
yticks(round([0, max(s_fr(:))], 2));

% true spike train
hold('on');
resp = get_resp(session, channel, unit, fold);
resp = mean(resp(trials, :), 1);
resp = reshape(resp, 1, []);
resp = [zeros(size(resp));s * resp];

times = reshape(times, 1, []);
times = [times;times];

plot(times, resp, ...
    'LineWidth', 3, ...
    'Color', colors(2, :));
hold('off');

% title(sprintf('Response of S-Model for trial #%d', trl));
if isscalar(trials)
    trials_txt = sprintf('trial #%s', type_trials);
else
    trials_txt = sprintf('%s trials', type_trials);
end
title(sprintf(...
    'Response of S-Model for %s \nDelta log-likelihood: %g', ...
    trials_txt, ...
    get_dll(session, channel, unit, fold, type_trials)));
xlabel('Time from saccade onset (ms)');
ylabel('Firing rate (spk/s)');

set_axis();

save_figure(...
    session, channel, unit, ...
    sprintf('s_fr_%s_fold%02d', type_trials, fold));
end
