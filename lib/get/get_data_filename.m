function filename = get_data_filename(session, channel, unit)
    % Get neuron's filename
    %
    % Parameters
    % ----------
    %
    % Returns
    % -------
    % - filename: char vector
    %   Neuron's filelname based on given `session` and `channel`

    info = get_info();
    
    filename = fullfile(...
        info.folders.data, ...
        get_filename(session, channel, unit));
end