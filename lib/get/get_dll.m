function dll = get_dll(session, channel, unit, fold, type_trials, type_times)
    % Get true delta log-likelihood of specific neuron
    %
    % Parameters
    % ----------
    %
    % - fold: integer scalar
    %   Fold number
    % - trials: vector
    %   Trial indeces
    % - times: vector
    %   Time of study
    %
    % Returns
    % -------
    % - dll: scalar
    %   Delta log-likelihood

    if nargin < 4
        fold = 1;
    end
    
    if nargin < 5
        type_trials = 'test';
    end
    
    if nargin < 6
        type_times = 'all';
    end
    
    r = get_resp(session, channel, unit, fold, type_trials, type_times);

    fr = get_s_fr(session, channel, unit, fold, type_trials, type_times);
    fr = fr ./ 1000;

    fr0 = get_fr0(session, channel, unit, fold, 'train, validation', 'all');
    fr0 = fr0 ./ 1000;

    dll = sum(r .* log(fr ./ fr0 + epsilon) - fr + fr0, 'all') / sum(r, 'all');
end