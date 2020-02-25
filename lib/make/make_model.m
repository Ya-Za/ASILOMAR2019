function make_model(session, channel, unit)
    % Make neural model
    %
    % Parameters
    % ----------
    % - session: scalar
    %   Session number with format: yymmdd
    % - channel: scalar
    %   Channel number

    info = get_info();
    
    times = info.times;
    num_folds = info.num_folds;

    neuron = load_profile(session, channel, unit);

    % create neural profile
    % todo:
    % - `rage_of_study` -> `times`
    % - `set_of_data` -> `data`
    % - `set_of_basis` -> `bases`
    nProfile = struct(...
        'session', session, ...
        'channel', channel, ...
        'unit', unit, ...
        'range_of_study', times, ...
        'set_of_data', load_data(session, channel, unit), ...
        'agenda', {make_agenda(neuron.set_of_B)}, ...
        'set_of_basis', make_bases(neuron.set_of_B));
    
    % cross validation
    for fold = 1:num_folds
        % train, validation, and test trials
        nProfile.set_of_trials = make_trials(neuron.set_of_trials, fold);

        % fit model
        fit_model(nProfile, session, channel, unit, fold);
    end
end

function agenda = make_agenda(stim_bases)
    % Make agenda of kernel names such as `stm`, `psk` and `off`
    %
    % Returns
    % -------
    % - agenda: cell array of char vectors
    %   Includes `stm1`, ..., `stm81`, `psk`, and `off`

    info = get_info();
    width = info.width;
    height = info.height;

    % n = width * height;
    % agenda = cell(n + 2, 1);

    % stimulus (stm)
%     for i = 1:n
%         agenda{i} = sprintf('stm%d', i);
%     end

    agenda = {};
    sz = [width, height];
    for x = 1:width
        for y = 1:width
            prb_ind = sub2ind(sz, x, y);
            B = stim_bases.(sprintf('B%d', prb_ind));
            if ~isempty(B)
                agenda{end + 1} = sprintf('stm%d', prb_ind);
            end
        end
    end

    % post-spike (psk)
    agenda{end + 1} = 'psk';

    % offset (off)
    agenda{end + 1} = 'off';
end

function bases = make_bases(stim_bases)
    % Make bases

    info = get_info();
    width = info.width;
    height = info.height;
    num_delays = info.num_delays;
    num_times = info.num_times;

    bases = struct();

    % stimulus (stm)
%     for i = 1:(width * height)
%         bases.(sprintf('stm%d', i)) = struct(...
%             'B', stim_bases.(sprintf('B%d', i)), ... % todo: stm1 -> stm(1)
%             'grd', 1, ...
%             'pow', 1);
%     end
    
    sz = [width, height];
    for x = 1:width
        for y = 1:width
            prb_ind = sub2ind(sz, x, y);
            
            B = stim_bases.(sprintf('B%d', prb_ind));
            if ~isempty(B)
                bases.(sprintf('stm%d', prb_ind)) = struct(...
                    'B', B, ... % todo: stm1 -> stm(1)
                    'grd', 1, ...
                    'pow', 1);
            end
        end
    end

    % post-spike (psk)
    % - delay bases
    delay_values = 1:num_delays;
    delay_knots = info.delay_knots;
    
    delay_bases = make_bsplie_bases(delay_values, delay_knots);

    % - time bases
    time_values = 1:num_times;
    time_bases = ones(1, length(time_values));

    % - time-value bases
    time_delay_bases = nan(size(time_bases, 1)*size(delay_bases, 1), size(time_bases, 2), size(delay_bases, 2));
    for t = 1:size(time_bases, 1)
        for d = 1:size(delay_bases, 1)
            prb_ind = sub2ind([size(time_bases, 1) size(delay_bases, 1)], t, d);

            time_delay_bases(prb_ind, :, :) = ...
                repmat(time_bases(t, :), size(delay_bases, 2), 1)'.*...
                repmat(delay_bases(d, :), size(time_bases, 2), 1);
        end
    end

    % todo: why `time_delay_bases` is 2d and not 1d?!
    bases.psk = struct(...
        'B', time_delay_bases, ...
        'grd', -1, ...
        'pow', 2);

    % offset (off)
    % - time bases
    res_t = 30;
    knots_t = ...
        min(time_values) - 2 * fix(res_t / 2):...
        fix(res_t/2):...
        2 * fix(res_t / 2) + max(time_values);

    time_bases = make_bsplie_bases(time_values, knots_t);

    bases.off = struct(...
        'B', time_bases, ...
        'grd', 1, ...
        'pow', 1);
end

function trials = make_trials(trials, fold)
    % Make trials

    trials = struct(...
        'trn_indices', trials.train_indices(fold).set, ...
        'trn_condtns', trials.train_conds(fold).set, ...
        'tst_indices', trials.test_indices(fold).set, ...
        'tst_condtns', trials.test_conds(fold).set, ...
        'crs_indices', trials.cross_indices(fold).set, ...
        'crs_condtns', trials.cross_conds(fold).set);
end
