function resp = get_resp(session, channel, unit, fold, type_trials, type_times)
    % Get true responses of specific neuron
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
    %   (trial x time) response
    
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
    resp = result.resp;
    
    trials = get_trials(session, channel, unit, fold, type_trials);
    times = get_times(type_times);
    
    resp = resp(trials, times);
end