function stim = code2stim(stimcode)
    % Convert stimcode to stim
    %
    % Parameters
    % ----------
    % - stimcode: integer matrix(trial, time)
    %   Coded stimuli
    %
    % Returns
    % -------
    % - stim: boolean array(trial, width, height, time)
    %   Stimulus

    info = get_info();
    width = info.width;
    height = info.height;

    % nl: Number of trials
    % nt: Number of times
    [nl, nt] = size(stimcode); % trial x time

    stim = zeros(nl, width, height, nt);

    sz = [width, height];
    for il = 1:nl
        for it = 1:nt
            ind = stimcode(il, it);

            if ind
                [x, y] = ind2sub(sz, ind);
                stim(il, x, y, it) = 1;
            end
        end
    end
end
