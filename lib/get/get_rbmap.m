function rbmap = get_rbmap(session, channel, unit)
    info = get_info();
    num_times = info.num_times;
    num_delays = info.num_delays;
    probe_time_resolution = info.probe_time_resolution;
    
    % smm
    smm = get_smm(session, channel, unit);
    
    % map
    sz = floor([num_times, num_delays] / probe_time_resolution) + 2;
    rbmap = smm.getRBMap(sz);
end