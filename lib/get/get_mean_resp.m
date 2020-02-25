function [mean_resp, se_resp, mean_s_fr, se_s_fr] = get_mean_resp(session, channel, unit, fold, period, probe, window)
% Get mean response and mean firing rate of s-model for a specific neuron
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
% - period: [scalar, scalar]
%   Time period from saccade
% - probe: 2-by-1 integer vector
%   [x, y] postion of a probe
% - window: scalar
%   Length of response window
%
% Returns
% -------
% - mean_resp: vector
%   Mean of responses
% - se_resp: vector
%   Standard error of responses
% - mean_s_fr: vector
%   Mean of s-model firing rate
% - se_s_fr: vector
%   Standard error of s-model firing rate

window = window - 1;

info = get_info();
width = info.width;
height = info.height;

% probe
probe = sub2ind([width, height], probe(1), probe(2));

% stimulus
stim = get_stim(session, channel, unit, fold);
stim = stim == probe;
[N, T] = size(stim); % N: trials, T: times

% period
tsac = ceil(T / 2); % saccade time
times = (period(1):period(2)) + tsac;

% response
resp = get_resp(session, channel, unit, fold);
% s-model firing rate
s_fr = get_s_fr(session, channel, unit, fold);

% mean response
resp_set = [];
s_fr_set = [];
for i = 1:N
    for t = times
        % if stim(i, t) == 0 && stim(i, t + 1) == 1
        if stim(i, t) == 1
            resp_set(end + 1, :) = resp(i, t:t + window);
            s_fr_set(end + 1, :) = s_fr(i, t:t + window);
        end
    end
end

resp_set = 1000 * resp_set;

% number of instances
n = size(resp_set, 1);

mean_resp = mean(resp_set);
mean_s_fr = mean(s_fr_set);

se_resp = std(resp_set) / sqrt(n);
se_s_fr = std(s_fr_set) / sqrt(n);
end
