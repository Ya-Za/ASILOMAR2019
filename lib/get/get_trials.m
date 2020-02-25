function trials = get_trials(session, channel, unit, fold, type)
    % Get trials of specific neuron
    %
    % Parameters
    % ----------
    %
    % - fold: integer scalar
    %   Fold number
    % - type: char vector
    % - Contains 'trn = 1', 'crs = 2', or 'tst = 3'
    %
    % Returns
    % -------
    % - trials: vector
    %   Set of trials

    if isnumeric(type)
        type = num2str(type);
    end
    
    result = load_result(session, channel, unit, fold);

    trials = [];

    % train
    if contains(type, {'trn', 'train', '1', 'all'})
        trials = [trials, result.trials.trn_indices];
    end
    % validation
    if contains(type, {'crs', 'val', 'validation', '2', 'all'})
        trials = [trials, result.trials.crs_indices];
    end
    % test
    if contains(type, {'tst', 'test', '3', 'all'})
        trials = [trials, result.trials.tst_indices];
    end
end