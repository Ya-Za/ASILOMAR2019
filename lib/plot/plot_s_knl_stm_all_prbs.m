function plot_s_knl_stm_all_prbs(session, channel, unit, fold)
    % Plot stimulus kernel of specific s-model for all probes
    %
    % Parameters
    % ----------
    %
    % - fold: integer scalar
    %   Fold number
    
    info = get_info();
    width = info.width;
    height = info.height;
    
    stm = get_s_knls(session, channel, unit, fold);

    create_figure('S-Model - Stimulus Kernel - All Porobes');

    for x = 1:width
        for y = 1:height
            % time-delay map
            ax = subplot(width, height, get_index(width, height, x, y));
            
            map = squeeze(stm(x, y, :, :));
            
            plot_time_delay_map(map);
            
            title(sprintf('(%d, %d)', x, y));
            xticks([]);
            yticks([]);
            xlabel('');
            ylabel('');
            colorbar('off');
            
            set(ax, 'FontSize', 8);
        end
    end
    
    ax = subplot(width, height, get_index(width, height, 1, 1));
    xlabel('Delay');
    ylabel('Time');

    suptitle(sprintf('Stimulus kernel of S-Model for all probes (fold #%d)', fold));

    save_figure(...
        session, channel, unit, ...
        sprintf('s_knl_stm_fold%02d', fold));
end

function index = get_index(width, height, x, y)
    r = height - y + 1;
    c = x;
    index = (r - 1) * width + c;
end