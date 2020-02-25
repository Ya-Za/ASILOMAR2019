function smm = get_smm(session, channel, unit)
    % SMM
    info = get_info();
    probe_time_resolution = info.probe_time_resolution;
    window = probe_time_resolution;
    times = info.times();
    tmin = times(1);
    tmax = times(end);
    width = info.width;
    height = info.height;
    threshold = info.smm_threshold;

    num_delays = info.num_delays;
    dmax = num_delays - 1;
    
    % - filename
    filename = get_data_filename(session, channel, unit);

    % - assets
    assets = fullfile(info.folders.assets, 'smm');

    % - props
    props = struct(...
        'procedureName', 'z', ...
        'window', window, ...
        'threshold', threshold, ...
        'erodeR', 0, ...
        'dilateR', 0, ...
        'tmin', tmin, ...
        'tmax', tmax, ...
        'dmax', dmax, ...
        'width', width, ...
        'height', height, ...
        'flags', struct('mask', false, 'test', 'no'));

    % - smm
    smm = SMM(filename, assets, props);
end