function knots = make_knots(values, delta)
    % Make knots
    %
    % Parameters
    % ----------
    % - values: vector
    %   Value of data points
    % - delta: scalar
    %   Distance between two data points
    %
    % Returns
    % -------
    % - knots: vector
    %   Knots

    half_delta = fix(delta / 2);
    delta = 2 * half_delta;

    knots = ...
        min(values) - delta:...
        half_delta:...
        max(values) + delta;
end