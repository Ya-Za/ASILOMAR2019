function trials = load_trials(session, channel, unit)
    % Make trials

    info = get_info();
    
    for fold = 1:info.num_folds
        fullfilename = fullfile(...
            info.folders.trials, ...
            get_id(session, channel, unit), ...
            sprintf('fold%02d.mat', fold));

        file = load_file(fullfilename);
        
        trials.train_indices(fold).set = file.set_of_trials.trn_indices;
        trials.train_conds(fold).set = file.set_of_trials.trn_condtns;
        trials.test_indices(fold).set = file.set_of_trials.tst_indices;
        trials.test_conds(fold).set = file.set_of_trials.tst_condtns;
        trials.cross_indices(fold).set = file.set_of_trials.crs_indices;
        trials.cross_conds(fold).set = file.set_of_trials.crs_condtns;
    end
    
end

function [trials] = load_trials_old(session, channel, unit)
    % Load seperated trials for train/validation/test data sets 
    %
    % Parameters
    % ----------
    % - session: scalar
    %   Session number with format: yymmdd
    % - channel: scalar
    %   Channel number
    % 
    % Returns
    % -------
    % - trials: struct (need more details)
    %   Set of trials for train/validation/test data sets

    info = get_info();

    fullfilename = fullfile(...
        info.folders.data, ...
        get_filename(session, channel, unit));

    file = load_file(fullfilename);
    
    % non-empty condition labels
    resp = double(file.resp);
    resp = sum(resp, 2);
    
    cond = double(file.cond);
    cond = cond(resp > 0);
    
    trials = get_trials(cond);
end

function trials = get_trials(cond)
    % Make trials

    info = get_info();
    num_folds = info.num_folds;
    pct_test = info.pct_test;
    pct_train = info.pct_train;
    pct_val = 1 - (pct_test + pc_train);
    
    ucond = unique(cond);
    trials = 1:numel(cond);
    
    for fold = 1:num_folds
        [train_cond, val_cond, test_cond] = dividerand(ucond, pct_train, pct_val, pct_test);
        
        train_idx = ismember(cond, train_cond);
        test_idx = ismember(cond, test_cond);
        val_idx = ismember(cond, val_cond);
        
        trials.train_indices(fold).set = trials(train_idx);
        trials.train_conds(fold).set = cond(train_idx);
        trials.test_indices(fold).set = trials(test_idx);
        trials.test_conds(fold).set = cond(test_idx);
        trials.cross_indices(fold).set = trials(val_idx);
        trials.cross_conds(fold).set = cond(val_idx);
    end
    
end

function trials = get_trials_old(all_conds)
    % Get trials
    %
    % Parameters
    % ----------
    % - all_conds: vector
    %   All nonempty condition labels
    %
    % Returns
    % -------
    % - trials: struct
    %   Trials for train/val/test data sets

    info = get_info();
    num_folds = info.num_folds;
    pct_test = info.pct_test;
    pct_train = info.pct_train;

    all_indices = 1:numel(all_conds);

    % test
    unique_conds = unique(all_conds);
    num_unique_conds = numel(unique_conds);
    test_conds_idx = randperm(num_unique_conds, fix(num_unique_conds * pct_test));

    [test_indices, test_conds] = make_correspond(unique_conds(test_conds_idx), all_conds, all_indices);

    trials.test_indices(1).set = test_indices;
    trials.test_conds(1).set = test_conds;

    % train/validation
    train_val_conds = unique_conds(setdiff(unique_conds, test_conds_idx));
    pct_train = pct_train / (1 - pct_test);

    for i = 1:num_folds
        % train
        train_conds_idx = randperm(numel(train_val_conds), fix(numel(train_val_conds) * pct_train));
        [train_indices, train_conds] = make_correspond(train_val_conds(train_conds_idx), all_conds, all_indices);
        trials.train_indices(i).set = train_indices;
        trials.train_conds(i).set = train_conds;

        % val
        val_conds_idx = setdiff(1:numel(train_val_conds), train_conds_idx);
        [val_indices, val_conds] = make_correspond(train_val_conds(val_conds_idx), all_conds, all_indices);
        trials.cross_indices(i).set = val_indices;
        trials.cross_conds(i).set = val_conds;


        if sum(abs(sort([trials.train_indices(i).set, trials.test_indices(1).set, trials.cross_indices(i).set])-all_indices)) ~= 0
            keyboard
        end
        if ...
                sum(diff(sort([trials.train_indices(i).set, trials.test_indices(1).set]))==0) ~= 0 || ...
                sum(diff(sort([trials.train_indices(i).set, trials.cross_indices(i).set]))==0) ~= 0 || ...
                sum(diff(sort([trials.test_indices(1).set, trials.cross_indices(i).set]))==0) ~= 0 || ...
                sum(diff(sort([unique(trials.train_conds(i).set), unique(trials.cross_conds(i).set)]))==0) ~= 0 || ...
                sum(diff(sort([unique(trials.train_conds(i).set), unique(trials.test_conds(1).set)]))==0) ~= 0 || ...
                sum(diff(sort([unique(trials.test_conds(1).set), unique(trials.cross_conds(i).set)]))==0)
            keyboard
        end
    end
end

function [indices, conds] = make_correspond(some_conds, all_conds, all_indices)
    conds = [];
    indices = [];
    for index = 1:numel(some_conds)
        c = some_conds(index);
        idx = (all_conds == c);
        conds = [conds, c * ones(1, sum(idx))];
        indices = [indices, all_indices(idx)];
    end
end