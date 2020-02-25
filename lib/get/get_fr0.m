function fr0 = get_fr0(session, channel, unit, fold, type_trials, type_times)
    % Get firing rate of null model
    %
    % Parameters
    % ----------
    %
    % - fold: integer scalar
    %   Fold number
    %
    % Returns
    % -------
    % - fr0: scalar
    %   null firing rate
    
    if nargin < 4
        fold = 1;
    end
    
    if nargin < 5
        type_trials = 'train, validation';
    end
    
    if nargin < 6
        type_times = 'all';
    end

    r = get_resp(session, channel, unit, fold, type_trials, type_times);
    fr0 = mean(r, 'all') * 1000;
end

function fr0 = get_fr0_old(session, channel, unit, fold)
    % Get firing rate of null model
    %
    % Parameters
    % ----------
    %
    % - fold: integer scalar
    %   Fold number
    %
    % Returns
    % -------
    % - fr0: scalar
    %   null firing rate
    
    % todo: notfair: by this definition the null model is as bad as smodel
    % without stimulus kernel (i.e. fr = smodel(s), if s = 0)

    [a, b] = get_nonlin_params(session, channel, unit, fold);

    [~, psk, off] = get_s_knls(session, channel, unit, fold);

    r = get_resp(session, channel, unit, fold);

    fr0 = zeros(size(r));
    for i = 1:size(r, 1)
        fr0(i, :) = a * logsig(conv(r(i, :), psk, 'same') + off + b);
    end
end
