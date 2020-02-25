function times = get_times(type, flag_reduced)
    % Get specific time period
    %
    % Parameters
    % ----------
    % - type: char vector
    % - Contains 'fix = 1', 'sac = 2', or 'all = 3'
    %
    % Returns
    % -------
    % - times: vector
    %   Set of time points

    if nargin < 1
        type = 'all';
    end
    
    if nargin < 2
        flag_reduced = false;
    end
    
    if isnumeric(type)
        type = num2str(type);
    end

    info = get_info();
    tmin = info.times(1);
    
    % fix
    if contains(type, {'fix', '1'})
        times = info.fix - tmin + 1;
    end
    % sac
    if contains(type, {'sac', '2'})
        times = info.sac - tmin + 1;
    end
    % all
    if contains(type, {'all', '3'})
        times = info.times - tmin + 1;
    end
    
    % reduced
    if flag_reduced
        times = unique(ceil(times ./ info.probe_time_resolution));
    end
end