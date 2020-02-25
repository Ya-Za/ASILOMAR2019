function [map, delay_bases, time_bases] = decide_bases_fast(session, channel, sigma)
    % Decide about resolution of bases
    %
    % Parameters
    % ----------
    % - session: scalar
    %   Session number with format: yymmdd
    % - channel: scalar
    %   Channel number
    % - sigma: scalar
    %   Scale standard deviation
    %   
    % Returns
    % -------
    % - map: array(width, height, time basis, delay basis)
    %   Map of bases for each probe
    % - delay_bases: matrix(delay basis, knot)
    %   Delay bases
    % - time_bases: matrix(time basis, knot)
    %   Time bases

    info = get_info();
    width = info.width;
    height = info.height;

    % time/delay bases
    bases = make_bases();
    time_bases = bases.B_t;
    delay_bases = bases.B_d;

    % map
    map = nan(width, height, size(time_bases, 1), size(delay_bases, 1));
    sz = [width, height];
    for x = 1:width
        for y = 1:height
            prb = sub2ind(sz, x, y);

            prb_map = load_map(session, channel, prb);

            map(x, y, :, :) = prb_map;
        end
    end
end