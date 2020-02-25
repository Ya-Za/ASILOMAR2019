function make_profile(session, channel, unit)
    % Make neural profiles
    %
    % Parameters
    % ----------
    % - session: scalar
    %   Session number with format: yymmdd
    % - channel: scalar
    %   Channel number

    info = get_info();
    
    profile_filename = fullfile(...
        info.folders.profiles, ...
        get_filename(session, channel, unit));
    
    if isfile(profile_filename)
        return
    end
    
    width = info.width;
    height = info.height;
    
    sigma = 1.5;

    set_of_trials = load_trials(session, channel, unit);

    save_timer = tic();

    % todo:
    % - `set_of_trials` -> `trials`
    profile = struct(...
        'session', session, ...
        'channel', channel, ...
        'unit', unit, ...
        'set_of_trials', set_of_trials); % todo: must be removed

    % basis function resolution
    [decided_map, B_d, B_t] = decide_bases(session, channel, unit, sigma);

    sz = [width, height];
    for x = 1:width
        for y = 1:height
            index = sub2ind(sz, x, y);

            t_map = squeeze(decided_map(x, y, :, :));
            idx_significant_bases = find(t_map == 1);
            num_significant_bases = length(idx_significant_bases);
            B = nan(num_significant_bases, size(B_t, 2), size(B_d, 2));

            for int_B = 1:length(idx_significant_bases)
                [ip, jp] = ind2sub([size(t_map, 1), size(t_map, 2)], idx_significant_bases(int_B));
                B(int_B, :, :) = repmat(B_t(ip, :), size(B_d, 2), 1)' .* repmat(B_d(jp, :), size(B_t, 2), 1);
            end

            profile.set_of_B.(sprintf('B%d', index)) = B;
        end
    end

    if ~exist(info.folders.profiles, 'dir')
        mkdir(info.folders.profiles);
    end

    fprintf('Save `%s`: ', profile_filename);
    save(profile_filename, '-struct', 'profile', '-v7.3');
    toc(save_timer);
end
