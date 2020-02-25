function select_bases_fast(session, channel, unit)
    % Select bases
    %
    % Parameters
    % ----------
    % - session: scalar
    %   Session number with format: yymmdd
    % - channel: scalar
    %   Channel number

    % p: probability of success/spike

    alpha = 0.01; % significanse level
    sw = 15; % smoothing window

    info = get_info();
    num_delays = info.num_delays;
    num_times = info.num_times;
    times = info.times;
    bases_folder = info.folders.bases;

    % building maps
    id = get_id(session, channel, unit);
    bases_subfolder = fullfile(bases_folder, id);
    if ~isfolder(bases_subfolder)
        mkdir(bases_subfolder);
    end

    % stim/resp
    times = (times(1) - num_delays):times(end);
    [stim, resp] = load_aligned_stim_resp(session, channel, times);
    resp = smoothdata(resp, 2, 'gaussian', sw);

    % structuring elements for erosion and dilation
    se_erode = strel('disk', 2);
    se_dilate = strel('disk', 14);

    % time/delay bases
    bases = make_bases();
    num_time_bases = size(bases.B_t, 1); % number of bases for time
    num_delay_bases = size(bases.B_d, 1); % number of bases for delay


    for probe = 1:(info.width * info.height)
        map_filename = fullfile(...
            bases_subfolder, ...
            sprintf('prb%02d.mat', probe));

        if ~isfile(map_filename)
            save_timer = tic();

            % time/delay map of basis functions
            map = false(num_times, num_delays);

            for it = 1:num_times % index of time
                t = it + num_delays;
                for d = 1:num_delays
                    idx = stim(:, t - d) == probe;

                    pref = resp(idx, t);
                    npref = resp(~idx, t);
                    if ~isempty(pref) && ~isempty(npref)
                        p = ptest(...
                            sum(pref), ...
                            sum(npref), ...
                            numel(pref), ...
                            numel(npref));

                        map(it, d) = p <= alpha;
                    end
                end
            end

            % clean and resize
            map = imerode(map, se_erode);
            map = imdilate(map, se_dilate);
            map = imresize(map, [num_time_bases, num_delay_bases]);

            fprintf('Save `%s`: ', map_filename);
            save(map_filename, 'map');
            toc(save_timer);
        end
    end
end

function [stim, resp] = load_aligned_stim_resp(session, channel, times)
    % Load aligned stimuli/responses with respect to the saccade onset
    
    info = get_info();

    fullfilename = fullfile(...
        info.folders.data, ...
        get_filename(session, channel, unit));

    file = load_file(fullfilename);

    STIM = double(file.stimcode);
    RESP = double(file.resp);

    % saccade aligned
    tsac = double(file.tsaccade);

    nl = numel(tsac); % number of trials
    nt = numel(times); % number of times

    stim = zeros(nl, nt);
    resp = zeros(nl, nt);

    for il = 1:nl
        it = times + tsac(il);

        stim(il, :) = STIM(il, it);
        resp(il, :) = RESP(il, it);
    end
end

function pv = ptest(x1, x2, n1, n2)
p1 = x1 / n1;
p2 = x2 / n2;
p = (x1 + x2) / (n1 + n2);
z = (p1 - p2) / sqrt(p * (1 - p) * ((1 / n1) + (1 / n2)));

pv = 2 * (1 - normcdf(abs(z)));
end
