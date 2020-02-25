function plot_results(session, channel, unit)
    % Plot some results
    %
    % Parameters
    % ----------
    %

    info = get_info();

    fprintf('===== Plot some resutls =====\n\n');
    main_timer = tic();

    probe = [7, 6]; % probe
    trials_type = 'test'; % 'train', 'test', 'validation' | scalar
    period = [-400 -100]; % time period from saccade
    window = info.num_delays; % length of response window

    for fold = 1:info.num_folds % for each fold
        % plot
        % - stimulus kernel of s-model
        plot_s_knl_stm(session, channel, unit, fold, probe);
        plot_s_knl_stm(session, channel, unit, fold);

        % % - estimated stimulus kernel with f-model
        % % plot_f_knl_stm(session, channel, unit, fold, probe);
        % % - average of estimated stimulus kernels with a-model
        % % plot_a_knl_stm(session, channel, unit, probe);

        % - response firing rate of s-model
        plot_s_fr(session, channel, unit, fold, trials_type);
        % - mean response
        plot_mean_resp(session, channel, unit, fold, period, probe, window);

        % - Kolmogorov Smirnov (KS) plot
        plot_ks(session, channel, unit, fold, period);
    end


    fprintf('\n\n');
    toc(main_timer);
end