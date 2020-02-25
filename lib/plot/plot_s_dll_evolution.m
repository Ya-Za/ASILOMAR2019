function plot_s_dll_evolution()
    % Plot delta log-likelihood evolution of trained s-model
    %
    % Parameters
    % ----------
    %

    t1 = 241;
    t2 = 541;
    
    info = get_info();
    num_times = info.num_times;
    times = info.times;
    tmin = times(1);
    tmax = times(end);
    
    [dll_evolution, ~, delta] = get_s_dll_evolution();
    
    create_figure('S-Model - DLL Evolution');
    
    xx = squeeze(nanmean(dll_evolution(:, :, :, 1), 2));
    subplot(223);
    
    plot(...
        (tmin + delta):(tmax - delta), ...
        squeeze(nanmedian(xx(:, (delta + 1):(num_times - delta), 1))));
    
    xlim([tmin, tmax]);
    
    lt1 = squeeze(nanmean(dll_evolution(:, :, t1, 1), 2));
    lt2 = squeeze(nanmean(dll_evolution(:, :, t2, 1), 2));
    
    xlabel('t to sacc');
    ylabel('bit/sp');
    title(['DLL evolution n = ' num2str(size(lt1, 1)) , ' win = ' num2str( 1 + 2 * delta)]);

    xticks(unique([tmin, t1 - tmin + 1, t2 - tmin + 1, 0, tmax]));
    set(gca(), 'XGrid', 'on');
    
    subplot(222);
    scatter(lt1, lt2, 40, '.k');
    hold('on');

    
    plot(linspace(-1, 1, 1000), median(lt2) * ones(1, 1000), '--r');
    plot(median(lt1) * ones(1,1000), linspace(-1, 1, 1000), '--r');
    plot(linspace(-1, 1, 1000), zeros(1,1000),'--b');
    plot(zeros(1, 1000), linspace(-1 , 1, 1000), '--b');
    xlabel('DLL fixation (bit/sp)');
    ylabel('DLL perisacc (bit/sp)');
    title(['DLL, pval = ' num2str(signrank(lt1,lt2))]);
    
    hold('off');

    subplot(221);
    q = histogram(lt2,20);
    hold('on');
    plot(median(lt2)*ones(1,1000),linspace(0,max(q.Values),1000),'--r');
    hold('off');
    
    title('perisacc DLL')


    subplot(224);
    q = histogram(lt1,20);
    hold('on');
    plot(median(lt1)*ones(1,1000),linspace(0,max(q.Values),1000),'--r');
    hold('off');
    title('fixation DLL')

    save_figure(...
        session, channel, unit, ...
        sprintf('s_dll_evolution_delta_%d', delta));
end
