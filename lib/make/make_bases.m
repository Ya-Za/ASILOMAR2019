function bases = make_bases()
    % Make basis functions
    %
    % Returns
    % -------
    % - bases: struct
    %   time/delay basis functions

    info = get_info();
    
    % stimulus bases
    % - delay
    delay_values = 1:info.num_delays;
    delay_delta = 2 * info.probe_time_resolution; % todo: why is twice than `7` ms
    delay_knots = make_knots(delay_values, delay_delta);
    delay_bases = make_bsplie_bases(delay_values, delay_knots);

    % - time
    time_values = 1:info.num_times;
    time_delta = 2 * info.probe_time_resolution;
    time_knots = make_knots(time_values, time_delta);
    time_bases = make_bsplie_bases(time_values, time_knots);

    % todo:
    % - `B_d` -> `delay`
    % - `B_t` -> `time`
    bases = struct(...
        'B_d', delay_bases, ...
        'B_t', time_bases);
end