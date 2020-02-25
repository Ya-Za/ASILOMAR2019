function id = get_id(session, channel, unit)
    % Get neuron's ID
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
    % - id: char vector
    %   Neuron's ID based on given `session` and `channel`

    id = sprintf('%d%02d%02d', session, channel, unit);
end