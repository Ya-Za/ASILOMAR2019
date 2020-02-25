function make_result(session, channel, unit)
    % Make result
    %
    % Parameters
    % ----------
    % - session: scalar
    %   Session number with format: yymmdd
    % - channel: scalar
    %   Channel number

    info = get_info();
    results_folder = info.folders.results;
    num_folds = info.num_folds;

    % todo: write `get_filename`
    id = get_id(session, channel, unit);

    results_model_folder = fullfile(results_folder, id);
    if ~exist(results_model_folder, 'dir')
        mkdir(results_model_folder);
    end

    for fold = 1:num_folds
        save_timer = tic();

        fold_name = sprintf('fold%02d.mat', fold);

        results_model_filename = fullfile(results_folder, id, fold_name);
        if ~isfile(results_model_filename)
            model = load_model(session, channel, unit, fold);

            % fork
            % - stm
            num_delays_stm = size(model.set_of_kernels.stm.knl, 4);
            fork_stm = nan(length(model.range_of_study), num_delays_stm);
            for counter = 1:num_delays_stm
                fork_stm(:, counter) = model.range_of_study - counter + 1;
            end
            % - psk
            num_delays_psk = size(model.set_of_kernels.psk.knl, 2);
            fork_psk = nan(length(model.range_of_study), num_delays_psk);
            for counter = 1:num_delays_psk
                fork_psk(:, counter) = model.range_of_study - counter + 1;
            end

            % Load data
            data = load_data(session, channel, unit);
            STIM = data.STIM;
            stimcode = data.stimcode;
            tsaccade = data.tsaccade;
            RESP = data.RESP;

            [num_trials, width, height, ~] = size(STIM);
            num_study_times = length(model.range_of_study);

            stim = nan(num_trials, num_study_times);
            resp = nan(num_trials, num_study_times);
            fr = nan(num_trials, num_study_times);
            fr_ss = nan(num_trials, num_study_times);

            % resp, and stim
            % todo: where `stim` and `resp` will be used
            for trial = 1:num_trials
                stim(trial, :) = stimcode(trial, tsaccade(trial) + model.range_of_study);
                resp(trial, :) = RESP(trial, tsaccade(trial) + model.range_of_study);
            end

            % data-driven, and self-sufficient
            for trial = 1:num_trials
                part_of_trials = zeros(1, size(fr, 2));

                for x = 1:width
                    for y = 1:height
                        part_of_stim = squeeze(STIM(trial, x, y, :));
                        comb = part_of_stim(max(fork_stm + tsaccade(trial), 1));
                        part_of_trials = ...
                            part_of_trials +...
                            sum(squeeze(model.set_of_kernels.stm.knl(x, y, :, :)) .* comb, 2)';
                    end
                end

                part_of_resp = RESP(trial, :);
                comb = part_of_resp(max(fork_psk + tsaccade(trial), 1));
                fr(trial, :) = phi(...
                    part_of_trials +...
                    model.set_of_kernels.off.knl +...
                    sum(model.set_of_kernels.psk.knl .* comb, 2)' +...
                    model.set_of_params.b0, ...
                    model.set_of_params.params);

                [for_model, ~] = make_resp(...
                    part_of_trials, ...
                    model.set_of_kernels.off.knl, ...
                    model.set_of_kernels.psk.knl, ...
                    model.set_of_params.params, ...
                    model.set_of_params.b0, ...
                    max(fork_psk - min(model.range_of_study) + 1, 1));

                fr_ss(trial, :) = for_model;
            end

            % saving data
            fprintf('Save `%s`: ', results_model_filename);
            trials = model.set_of_trials;

            save(results_model_filename, 'stim', 'resp', 'fr', 'fr_ss', 'trials');

            toc(save_timer);
        end
    end
end

function [n_lmbd, n_resp] = make_resp(sout, oout, h, param, b0, fork)
    % Make response firing rates

    n_lmbd = phi(sout + oout + b0, param);
    n_resp = zeros(size(n_lmbd));
    for tim_point = 2:length(n_lmbd)
        spk = rand(1) < n_lmbd(tim_point)/1000;
        if spk
            n_resp(tim_point) = 1;
            comb = n_resp(fork);
            temporary_lambda = phi(sout + oout + sum(h.*comb, 2)' + b0, param);
            n_lmbd(tim_point:end) = temporary_lambda(tim_point:end);
        end
    end
end
