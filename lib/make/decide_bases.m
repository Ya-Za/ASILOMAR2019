function [map, delay_bases, time_bases] = decide_bases(session, channel, unit, sigma)
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

    % time/delay bases
    bases = make_bases();
    delay_bases = bases.B_d;
    time_bases = bases.B_t;
    
    % map
    map = get_rbmap(session, channel, unit);
end

function [map, delay_bases, time_bases] = decide_bases_old(session, channel, unit, sigma)
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
    bases_iterations = info.bases_iterations;

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

            prb_map = load_map(session, channel, unit, prb);

            sig_mean = mean(prb_map(:, :, 1:bases_iterations), 3);
            ctl_mean = mean(prb_map(:, :, bases_iterations + 1:bases_iterations + bases_iterations), 3);
            sig_std = std(prb_map(:, :, bases_iterations + 1:bases_iterations + bases_iterations), [], 3);
            map(x, y, :, :) = abs(sig_mean - ctl_mean) >= sigma .* sig_std;
        end
    end
end
