function data = load_data(session, channel, unit)
    % Load data
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
    % - STIM: boolean array(trial, width, height, time)
    %   Stimuli
    % - stimcode: integer matrix(trial, time)
    %   Coded stimuli
    % - tsaccade: integer vector(trial)
    %   Saccade times
    % - RESP: boolean matrix(trial, time)
    %   Responses

    info = get_info();

    fullfilename = fullfile(...
        info.folders.data, ...
        get_filename(session, channel, unit));

    file = load_file(fullfilename);
    
    if ~isfield(file, 'tsac')
        [nl, nt] = size(file.resp); % number of trials/times
        file.tsac = ceil(nt / 2) * ones(nl, 1);
    end

    % todo:
    % - `STIM` -> `stim`
    % - `RESP` -> `resp`
    % - `tsaccade` -> `tsac`
    data = struct(...
        'STIM', code2stim(file.stim), ...
        'stimcode', double(file.stim), ...
        'tsaccade', double(file.tsac), ...
        'RESP', double(file.resp));
end
