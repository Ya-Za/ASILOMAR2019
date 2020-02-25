function fr = get_s_fr(session, channel, unit, fold, type_trials, type_times)
    % Get firing rate of specific s-model
    %
    % Parameters
    % ----------
    %
    % - fold: integer scalar
    %   Fold number
    %
    % Returns
    % -------
    % - resp: 2D matrix
    %   (trial x time) firing rate

    if nargin < 4
        fold = 1;
    end

    if nargin < 5
        type_trials = 'all';
    end

    if nargin < 6
        type_times = 'all';
    end


    result = load_result(session, channel, unit, fold);
    fr = result.fr;
    
    trials = get_trials(session, channel, unit, fold, type_trials);
    times = get_times(type_times);
    
    fr = fr(trials, times);
end