function filename = get_filename(session, channel, unit)
    % Get neuron's filename
    %
    % Parameters
    % ----------
    %
    % Returns
    % -------
    % - filename: char vector
    %   Neuron's filelname based on given `session` and `channel`

    id = get_id(session, channel, unit);
    filename = sprintf('%s.mat', id);
end