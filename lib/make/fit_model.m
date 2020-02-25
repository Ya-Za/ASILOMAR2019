function fit_model(profile, session, channel, unit, fold)
    % Fit neural model
    %
    % Parameters
    % ----------
    % - profile: struct
    %   Neural profile
    %
    % - fold: scalar
    %   Fold number

    info = get_info();
    
    % if exist neural model
    model_folder = fullfile(info.folders.models, get_id(session, channel, unit));
    model_filename = fullfile(model_folder, sprintf('fold%02d.mat', fold));
    
    if isfile(model_filename)
        return
    end
    
    bs_folder = fullfile(...
        info.folders.BS, ...
        get_id(session, channel, unit), ...
        sprintf('fold%02d', fold));
    
    if ~isfolder(bs_folder)
        mkdir(bs_folder);
    end

    save_timer = tic();

    % remove `test` data
    [profile, trn_indices, val_indices] = get_trn_val_indices(profile);

    % set of initialization
    profile.set_of_params = get_params(profile, session, channel, unit, fold);

    % alternating minimization (coordinate ascent)
    % termination condition: log-likelihood improvement < threshold (e.x. 0.01)
    for state = [false, true] % state of `offset` kernel in the process of learning
        th = 1e-2;
        do_continue = true;
        while do_continue % for each improvement
            do_continue = false;

            for i = 1:length(profile.agenda)
                agenda = profile.agenda{i};

                if contains(agenda, 'off') == state
                    bs_filename = fullfile(...
                        bs_folder, ...
                        [agenda, '.mat']);

                    fprintf('Load `%s`: ', bs_filename);

                    tic();
                    load(bs_filename, 'BS');
                    toc();

                    % before
                    % - train
                    train_before = loglike(...
                        profile.set_of_params.(agenda).x, ...
                        agenda, ...
                        trn_indices, ...
                        profile.agenda, ...
                        profile.set_of_params, ...
                        profile.set_of_basis.(agenda).grd, ...
                        profile.set_of_basis.(agenda).pow, ...
                        BS);
                    % - validation
                    val_before = loglike(...
                        profile.set_of_params.(agenda).x, ...
                        agenda, ...
                        val_indices, ...
                        profile.agenda, ...
                        profile.set_of_params, ...
                        profile.set_of_basis.(agenda).grd, ...
                        profile.set_of_basis.(agenda).pow, ...
                        BS);

                    improvement = inf;
                    learning_rate = 1e+3; % learning rate

                    while improvement > th && learning_rate > get_th(agenda)
                        % update parameters
                        train_grad = grad_comp(...
                            agenda, ...
                            trn_indices, ...
                            profile.agenda, ...
                            profile.set_of_params, ...
                            profile.set_of_basis, ...
                            BS);

                        x = profile.set_of_params.(agenda).x + ...
                            learning_rate .* train_grad;

                        % after
                        % - train
                        train_after = loglike(...
                            x, ...
                            agenda, ...
                            trn_indices, ...
                            profile.agenda, ...
                            profile.set_of_params, ...
                            profile.set_of_basis.(agenda).grd, ...
                            profile.set_of_basis.(agenda).pow, ...
                            BS);
                        % - validation
                        val_after = loglike(...
                            x, ...
                            agenda, ...
                            val_indices, ...
                            profile.agenda, ...
                            profile.set_of_params, ...
                            profile.set_of_basis.(agenda).grd, ...
                            profile.set_of_basis.(agenda).pow, ...
                            BS);

                        % no improvement
                        if ...
                                isnan(train_after) ||...
                                isnan(val_after) || ...
                                train_after <= train_before ||...
                                val_after <= val_before

                            learning_rate = learning_rate / 10;
                            disp([agenda, ', mu is upgraded from ', num2str(learning_rate * 10), ' to -> ', num2str(learning_rate)])
                        else
                            improvement = norm(learning_rate .* train_grad) ./...
                                (norm(x - learning_rate .* train_grad) + epsilon);

                            % update `x`
                            profile.set_of_params.(agenda).x = x;

                            profile.set_of_params.(agenda).val = ...
                                update_component_value(...
                                    BS, ...
                                    profile.set_of_params.(agenda).x, ...
                                    profile.set_of_basis.(agenda).grd, ...
                                    profile.set_of_basis.(agenda).pow);

                            disp([agenda, ', mu = ', num2str(learning_rate), ': ', num2str(train_after), ' vs. ', num2str(val_after), ' (', num2str(improvement), ')'])

                            try
                                new_iter = length(profile.set_of_iterations.iter) + 1;
                            catch
                                new_iter = 1;
                            end

                            profile.set_of_iterations.iter(new_iter) = struct(...
                                'lbl', agenda, ...
                                'mu', learning_rate, ...
                                'x', x, ...
                                'train_llhd', train_after, ...
                                'cross_llhd', val_after, ...
                                'alternating_improvement', improvement);

                            % save current values
                            train_before = train_after;
                            val_before = val_after;

                            % update termination condition
                            do_continue = do_continue | (improvement > th);
                        end % if
                    end % while

                    clear('BS');
                end % if
            end % for
        end % while
    end % for
    
    % set of kernels
    profile.set_of_kernels = make_kernels(...
        profile.agenda, ...
        profile.set_of_params, ...
        profile.set_of_basis);

    % save neural profile
    if ~isfolder(model_folder)
        mkdir(model_folder);
    end

    fprintf('Save `%s`: ', model_filename);

    % remove-fields
    profile = rmfield(profile, {'set_of_basis', 'set_of_data'});

    save(model_filename, '-struct', 'profile', '-v7.3')
    toc(save_timer);
    
    % remove `BS` folder
    rmdir(bs_folder, 's');
end

function [nProfile, trn_indices, val_indices] = get_trn_val_indices(nProfile)
    % Get train/validation indeces

    % - To relase memory, remove test data and just keep `train`, and 
    % `validation` trials

    num_trials = size(nProfile.set_of_data.STIM, 1);
    % - 0: no responses
    all_indices = zeros(1, num_trials);
    % - 1: train
    trn_indices = nProfile.set_of_trials.trn_indices;
    all_indices(trn_indices) = 1; % train
    % - 2: validation
    val_indices = nProfile.set_of_trials.crs_indices; %todo: change `crs` to `val(idation)`
    all_indices(val_indices) = 2; % val
    % - 3: test
    tst_indices = nProfile.set_of_trials.tst_indices;
    all_indices(tst_indices) = 3; % test

    % update neural profile
    trn_val_indices = (all_indices == 1 | all_indices == 2);

    nProfile.set_of_data.STIM = nProfile.set_of_data.STIM(trn_val_indices, :, :, :);
    nProfile.set_of_data.stimcode = nProfile.set_of_data.stimcode(trn_val_indices, :);
    nProfile.set_of_data.tsaccade = nProfile.set_of_data.tsaccade(trn_val_indices);
    nProfile.set_of_data.RESP = nProfile.set_of_data.RESP(trn_val_indices, :);

    % indices on train/validation data (all data mins `0` and `3` labels)
    trn_val_indices = all_indices(trn_val_indices);
    trn_indices = find(trn_val_indices == 1);
    val_indices = find(trn_val_indices == 2);
end

function params = get_nonlin_params_old(nProfile)
    % Get parameters of non-linearity

    % smoothing window size
    win = 50;
    params = [
        get_max_fr(...
            nProfile.set_of_data.RESP, ...
            nProfile.set_of_data.stimcode, ...
            win) % todo: use constant `WINDOW`
        0]; % todo: `0` -> -`b0`
end

function params = get_nonlin_params(session, channel, unit)
    % Get parameters of non-linearity

    data = load_data(session, channel, unit);
    
    % smoothing window size
    win = 50;
    params = [
        get_max_fr(...
            data.RESP, ...
            data.stimcode, ...
            win) % todo: use constant `WINDOW`
        0]; % todo: `0` -> -`b0`
end

function params = get_nonlin_params2(session, channel, unit, fold)
    % Get parameters of non-linearity

    info = get_info();
    
    filename = fullfile(...
        [info.folders.models, '-old'], ...
        get_id(session, channel, unit), ...
        sprintf('fold%02d.mat', fold));
    
    load(filename, 'set_of_params');
    params = set_of_params.params;
end

function b0 = get_base_act_old(nProfile, params)
    % Get baseline activitys

    % todo: why does not `b0` update using `gradient ascent`?
    % `b0` is redundant and must be put in `params(2)`
    b0 = inv_phi(...
        get_mean_fr(...
            nProfile.set_of_data.RESP, ...
            nProfile.set_of_data.stimcode), ...
        params);
end

function b0 = get_base_act(session, channel, unit, params)
    % Get baseline activitys

    % todo: why does not `b0` update using `gradient ascent`?
    % `b0` is redundant and must be put in `params(2)`
    data = load_data(session, channel, unit);
    
    b0 = inv_phi(...
        get_mean_fr(...
            data.RESP, ...
            data.stimcode), ...
        params);
end

function b0 = get_base_act2(session, channel, unit, fold)
    % Get baseline activitys

    info = get_info();
    
    filename = fullfile(...
        [info.folders.models, '-old'], ...
        get_id(session, channel, unit), ...
        sprintf('fold%02d.mat', fold));
    
    load(filename, 'set_of_params');
    b0 = set_of_params.b0;
end

function max_firing_rate = get_max_fr(resp, stim, win)
    % Estimate maximum firing rate

    lambda = nan(size(resp));
    for trial = 1:size(resp, 1)
        % todo: I think lambda must be compute `vertically` (on trials) instead
        % `horizontally` (on time)
        
        % lambda(trial, :) = smoothdata(resp(trial, :), 'gaussian', win);
        lambda(trial, :) = gauss_smooth(resp(trial, :), win);
    end
    
    % max_firing_rate = 1000 * max(lambda(stim > 0));
    max_firing_rate = max(lambda(stim > 0));
end

function mean_firing_rate = get_mean_fr(resp, stim)
    % Estimate minimum firing rate

    mean_firing_rate = 1000 * mean(resp(stim > 0), 'all');
end

function set_of_params = get_params(nProfile, session, channel, unit, fold)
    % Get parameters of neural model

    info = get_info();
    bs_folder = fullfile(...
        info.folders.BS, ...
        get_id(session, channel, unit), ...
        sprintf('fold%02d', fold));

    set_of_params = struct();
    
    % set_of_params.params = get_nonlin_params(nProfile);
    % set_of_params.b0 = get_base_act(nProfile, set_of_params.params);
    
    set_of_params.params = get_nonlin_params(session, channel, unit);
    set_of_params.b0 = get_base_act(session, channel, unit, set_of_params.params);
    
    % set_of_params.params = get_nonlin_params(session, channel, unit, fold);
    % set_of_params.b0 = get_base_act(session, channel, unit, fold);

    for i = 1:length(nProfile.agenda)
        agenda = nProfile.agenda{i};

        % BS
        bs_filename = fullfile(bs_folder, [agenda, '.mat']);

        if ~exist(bs_filename, 'file')
            save_timer = tic();

            BS = make_BS(...
                nProfile.set_of_data, ...
                nProfile.set_of_basis, ...
                agenda, ...
                nProfile.range_of_study);

            fprintf('Save `%s`: ', bs_filename); 
            save(bs_filename, 'BS', '-v7.3');
            toc(save_timer);
        else
            fprintf('Load `%s`: ', bs_filename);
            tic();
            load(bs_filename, 'BS');
            toc();
        end

        % x := initial value of gradient ascent (weight of bases)
        set_of_params.(agenda).x = ...
            zeros(size(nProfile.set_of_basis.(agenda).B, 1), 1) +...
            1e-6;

        % signal
        set_of_params.(agenda).sig = make_sig(...
            nProfile.set_of_data, ...
            agenda, ...
            nProfile.range_of_study);

        % value
        set_of_params.(agenda).val = update_component_value(...
            BS, ...
            set_of_params.(agenda).x, ...
            nProfile.set_of_basis.(agenda).grd, ...
            nProfile.set_of_basis.(agenda).pow);
    end
end

function BS = make_BS(set_of_data, set_of_basis, agenda, range_of_study)
    % Make BS := (stm = B * S) | (psk = B * r) | (off = B)

    resp = set_of_data.RESP; % trial x time
    stim = set_of_data.STIM; % trail x probe_x x probe_y x time
    B = set_of_basis.(agenda).B; % basis x time x delay

    num_trials = size(resp, 1); % number of trials
    [num_basis, num_times, num_delays] = size(B); % number of bases, times, and delays        
    num_study_times = length(range_of_study); % number of study times

    % BS
    BS = nan(num_basis, num_trials, num_study_times); % basis x trial x study time

    switch agenda
        case 'psk'
            % BS
            % - indexes for convolution (dot product)
            fork_for_resp = nan(num_times, num_delays); % time x delay
            for i = 1:num_delays
                fork_for_resp(:, i) = range_of_study - i + 1;
            end

            for i = 1:num_trials
                part_of_resp = resp(i, :);
                part_of_resp(1) = 0;

                t_comb = nan(1, num_times, num_delays);
                t_comb(1, :, :) = part_of_resp(fork_for_resp + set_of_data.tsaccade(i));

                BS(:, i, :) = sum(B .* repmat(t_comb, num_basis, 1, 1), 3);
            end
        case 'off'
            % BS
            for i = 1:size(resp, 1)
                BS(:, i, :) = B;
            end
        otherwise % `stm`
            % BS
            fork_of_stim = nan(size(B, 2), size(B, 3));
            for i = 1:size(B, 3)
                fork_of_stim(:, i) = range_of_study - i + 1;
            end

            [x, y] = ind2sub([size(stim, 2), size(stim, 3)], str2double(agenda(4:end)));

            for i = 1:size(stim, 1)
                part_of_stim = squeeze(stim(i, x, y, :));
                t_comb = nan(1, size(B, 2), size(B, 3));
                t_comb(1, :, :) = part_of_stim(fork_of_stim + set_of_data.tsaccade(i));
                BS(:, i, :) = sum(B.*repmat(t_comb, size(B, 1), 1, 1), 3);
            end
    end
end

function sig = make_sig(set_of_data, agenda, range_of_study)
    % Make signal

    resp = set_of_data.RESP; % trial x time
    num_trials = size(resp, 1); % number of trials
    num_study_times = length(range_of_study); % number of study times

    % signal
    sig = [];

    switch agenda
        case 'psk'
            % signal
            sig = nan(num_trials, num_study_times); % trial x study time

            for i = 1:num_trials
                part_of_resp = resp(i, :);
                part_of_resp(1) = 0;

                sig(i, :) = part_of_resp(set_of_data.tsaccade(i) + range_of_study);
            end
    end
end

function out = update_component_value(BS, x, grd, pow)
    % Update values
    out = squeeze(sum((grd .* (x .^ pow)) .* BS, 1));
end

function grd = grad_comp(acode, indices, agenda, set_of_params, set_of_basis, BS)
    % Gradient of components

    delta = 1/1000;
    b0 = set_of_params.b0;
    params = set_of_params.params;
    rsp = set_of_params.psk.sig(indices, :);
    x = set_of_params.(acode).x;
    pow = set_of_basis.(acode).pow;
    cof = set_of_basis.(acode).grd;
    BS = BS(:, indices, :);
    inside_phi = zeros(size(BS, 2), size(BS, 3));
    for intenda = 1:length(agenda)
        inside_phi = inside_phi + set_of_params.(char(agenda(intenda))).val(indices, :);
    end
    lmd = phi(inside_phi+b0, params);

    BS2d = reshape(BS, [size(BS, 1), size(BS, 2)*size(BS, 3)]);
    ld2d = repmat(reshape(lmd, [1, size(BS, 2)*size(BS, 3)]), size(BS, 1), 1);
    rs2d = repmat(reshape(rsp, [1, size(BS, 2)*size(BS, 3)]), size(BS, 1), 1);
    grd = sum(((rs2d./ld2d)-delta) .* dphi(inv_phi(ld2d, params), params) .* ((cof.*pow.*(x.^(pow-1))).*BS2d) , 2);
end

function out = loglike(x, acode, indices, agenda, set_of_params, grd, pow, BS)
    % Loglikelihood

    delta = 1 / 1000;
    b0 = set_of_params.b0;
    params = set_of_params.params;
    rsp = set_of_params.psk.sig(indices, :);
    BS = BS(:, indices, :);
    inside_phi = zeros(size(BS, 2), size(BS, 3));
    for i = 1:length(agenda)
        if strcmp(char(agenda(i)), acode)
            inside_phi = inside_phi + update_component_value(BS, x, grd, pow);
        else
            inside_phi = inside_phi + set_of_params.(char(agenda(i))).val(indices, :);
        end
    end
    lmd = phi(inside_phi+b0, params);

    ll_over_trials = rsp .* log((lmd + epsilon) * delta) - ((lmd + epsilon) * delta);
    out = sum(sum(ll_over_trials));
end

function th = get_th(agenda)
    % Get threshold

    switch agenda
        case 'psk'
            th = 1e-12;
        case 'off' % todo: must be updated! 1e-20
            th = 1e-12;
        otherwise
            th = 1e-8;
    end
end

function set_of_kernels = make_kernels(agenda, set_of_params, set_of_basis)
    % Make kernels

    info = get_info();
    width = info.width;
    height = info.height;
    sz = [width, height];
    num_times = info.num_times;
    num_delays = info.num_delays;

    set_of_kernels = struct();
    knl = zeros(width, height, num_times, num_delays);
    for intenda = 1:length(agenda)
        acode = char(agenda(intenda));
        switch acode
            case 'psk'
                x = set_of_params.(acode).x;
                B = set_of_basis.(acode).B;
                pow = set_of_basis.(acode).pow;
                grd = set_of_basis.(acode).grd;
                set_of_kernels.(acode).knl = squeeze(sum((grd .* (x .^ pow)) .* B, 1));
            case 'off'
                set_of_kernels.(acode).knl = set_of_params.(acode).x' * set_of_basis.(acode).B;
            otherwise
                x = set_of_params.(acode).x;
                B = set_of_basis.(acode).B;
                pow = set_of_basis.(acode).pow;
                grd = set_of_basis.(acode).grd;
                [i, j] = ind2sub(sz, str2double(acode(4:end)));
                knl(i, j, :, :) = squeeze(sum((grd .* (x .^ pow)) .* B, 1));
        end
    end

    set_of_kernels.stm.knl = knl;
end
