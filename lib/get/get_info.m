function info = get_info()
    % Get global information
    %
    % Returns
    % -------
    % - info: struct
    %   - width: scaler
    %     Width of the probes' grid
    %   - height: scaler
    %     Height of the probes' grid
    %   - fix: vector
    %     Saccade aligned presaccadic times
    %   - sac: vector
    %     Saccade aligned perisaccadic times
    %   - times: vector
    %     Saccade aligned study times
    %   - num_times: scalar
    %     Number of study times
    %   - num_delays: scalar
    %     Number of delays
    %   - probe_time_resolution: scaler
    %     Duration that a probe is on
    %   - num_folds: scalar
    %     Number of folds
    %   - pct_test: scalar
    %     Percentage of test data
    %   - pct_train: scalar
    %     Percentage of train data
    %   - bases_iterations: scalar
    %     Number of iterations to find important bases
    %   - folders: struct
    %       - assets: char vector
    %         Path to the `assets` folder
    %       - data: char vector
    %         Path to the `data` folder
    %       - bases: char vector
    %         Path to the `bases` folder
    %       - BS: char vector
    %         Path to the `BS` folder
    %       - profiles: char vector
    %         Path to the `profiles` folder
    %       - models: char vector
    %         Path to the `models` folder
    %       - avg_models: char vector
    %         Path to the `avg-models` folder
    %       - results: char vector
    %         Path to the `results` folder
    
    % global smm_threshold
    smm_threshold = get_global('smm_threshold');
    
    width = 9;
    height = 9;
    times = -540:+540;
    fix = -375:-225; % -450:0;
    sac = -75:+75; % 0:150;
    num_times = length(times);
    num_delays = 200;
    probe_time_resolution = 7; % todo: `probe_time_resolution` -> `time_resolution`
    num_folds = 1;
    pct_test = 0.35; % todo: pct_validation
    pct_train = 0.35;
    bases_iterations = 100; % for each `true` and `shuffle`
    delay_knots = [1 2 3 4 6 8 15 22 29 36 43 50 57 64 71 78 92 106 120 134 148 162 176];
    
    % SMM
    % smm_threshold = 0.6;
    % Plotting
    plotting = struct(...
        'formattype', '.epsc');
    % F-Model
    foptions = optimset(...
        'Display', 'off', ...
        'MaxFunEvals', 1e3, ...
        'MaxIter', 1e3, ...
        'TolX', 1e-3, ...
        'TolFun', 1e-6, ...
        'TolCon', 1e-3);

    lags = [
        001 020
        020 040
        040 050
        050 053
        053 056
        056 059
        059 062
        062 065
        065 068
        068 071
        071 074
        074 077
        077 080
        080 085
        085 090
        090 095
        095 100
        100 105
        105 110
        110 115
        115 120
        120 125
        125 130
        130 135
        135 140
        140 145
        145 150];
    
    % Folders
    % assets_folder = 'assets';
    assets_folder = '/uufs/chpc.utah.edu/common/home/noudoost-group1/yasin/asimolar/all';
    smm_threshold_folder = sprintf('th-%.2f', smm_threshold);
    folders = struct(...
        'assets', assets_folder, ...
        'data', fullfile(assets_folder, 'data'), ...
        'trials', fullfile(assets_folder, 'trials'), ...
        'bases', fullfile(assets_folder, 'bases'), ...
        'BS', fullfile(assets_folder, 'BS', smm_threshold_folder), ...
        'profiles', fullfile(assets_folder, 'profiles', smm_threshold_folder), ...
        'models', fullfile(assets_folder, 'models', smm_threshold_folder), ...
        'avg_models', fullfile(assets_folder, 'avg-models', smm_threshold_folder), ...
        'results', fullfile(assets_folder, 'results', smm_threshold_folder));
    % Info
    info = struct(...
        'width', width, ...
        'height', height, ...
        'times', times, ...
        'fix', fix, ...
        'sac', sac, ...
        'num_times', num_times, ...
        'num_delays', num_delays, ...
        'probe_time_resolution', probe_time_resolution, ...
        'num_folds', num_folds, ...
        'pct_test', pct_test, ...
        'pct_train', pct_train, ...
        'bases_iterations', bases_iterations, ...
        'delay_knots', delay_knots, ...
        'smm_threshold', smm_threshold, ...
        'plotting', plotting, ...
        'foptions', foptions, ...
        'lags', lags, ...
        'folders', folders);
end