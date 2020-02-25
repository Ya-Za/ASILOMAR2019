function plot_s_dll_vs_ratios(thresholds)
    % Plot delta log-likelihood of trained s-models
    %
    % Parameters
    % ----------
    %
    
%     plot1(thresholds);
%     plot2(thresholds);
%     plot3(thresholds);
    plot4(thresholds);
%     plot5(thresholds);
end

% Plots
% each - dll
function plot1(thresholds)
    info = get_info();
    
    % data
    data = get_dll_vs_ratios(thresholds);

    % [threshold, neuron, trn|val|tst, fix|sac, fold]
    dll = mean(data.dll, 5); % mean along dimension `fold`
    ratios = mean(data.ratios, 5); % mean along dimension `fold`
    ids = data.ids;
    
    % plot
    % - each neuron
    for i = 1:numel(ids)
        
        id = ids{i};
        create_figure(sprintf('DLL vs Ratios - %s', id));
        hold('on');
        
        for type_trials = [1, 3] % ["train", "validation", "test"]
            for type_times = 1:2 % ["fix", "sac", "all"]
                
                plot_dll_vs_ratios_for_a_neuron(...
                    ratios(:, i, type_trials, type_times), ...
                    dll(:, i, type_trials, type_times), ...
                    get_display_name(type_trials, type_times));
                
        %         % threshold-axis
        %         ax = gca();
        %         axes('Position', ax.Position, ...
        %             'XAxisLocation', 'top', ...
        %             'Color', 'none');
            end
        end
        
        hold('off');
        lgd = legend();
        title(lgd, 'Trials, Times');
        title(id);
        xlabel('\rho = #spikes / #parameters');
        ylabel('\DeltaLL/spk (bit/sec)');

        set_axis();

        saveas(...
            gcf, ...
            fullfile(...
                info.folders.assets, ...
                'results', ...
                ['dll-', id, info.plotting.formattype]));
            
        close('all');
    end
    
    % Local functions
    function display_name = get_display_name(type_trials, type_times)
        name_trials = {'train', 'validation', 'test'};
        name_times = {'fix', 'sac', 'all'};
        
        display_name = sprintf('%s - %s', ...
            name_trials{type_trials}, ...
            name_times{type_times});
    end
    
    function plot_dll_vs_ratios_for_a_neuron(ratios, dll, name)
        line_width = 4;
        marker_size = 100;
        
        plot(ratios, dll, ...
            'DisplayName', name, ...
            'LineWidth', line_width, ...
            'LineStyle', '-', ...
            'Marker', '.', ...
            'MarkerSize', marker_size);
        
        set(gca(), 'XScale', 'log');
    end
end

% each - tst/trn
function plot2(thresholds)
    info = get_info();
    
    % data
    data = get_dll_vs_ratios(thresholds);

    % [threshold, neuron, trn|val|tst, fix|sac, fold]
    dll = mean(data.dll, 5); % mean along dimension `fold`
    ratios = mean(data.ratios, 5); % mean along dimension `fold`
    ids = data.ids;
    
    % plot
    % - each neuron
    for i = 1:numel(ids)
        
        id = ids{i};
        create_figure(sprintf('DLL vs Ratios - %s', id));
        hold('on');
        
        for type_times = 1:2 % ["fix", "sac", "all"]
            plot_dll_vs_ratios_for_a_neuron(...
                ratios(:, i, 1, type_times), ...
                dll(:, i, 3, type_times) ./ dll(:, i, 1, type_times), ...
                get_display_name(type_times));

    %         % threshold-axis
    %         ax = gca();
    %         axes('Position', ax.Position, ...
    %             'XAxisLocation', 'top', ...
    %             'Color', 'none');
        end
        
        hold('off');
        lgd = legend();
        title(lgd, 'Times');
        title(id);
        xlabel('\rho = #spikes / #parameters');
        ylabel('PR [test] / PR [train]');

        set_axis();

        saveas(...
            gcf, ...
            fullfile(...
                info.folders.assets, ...
                'results', ...
                ['dll-tst-trn-', id, info.plotting.formattype]));
            
        close('all');
    end
    
    % Local functions
    function display_name = get_display_name(type_times)
        name_times = {'fix', 'sac', 'all'};
        
        display_name = sprintf('%s', ...
            name_times{type_times});
    end

    function plot_dll_vs_ratios_for_a_neuron(ratios, dll, name)
        line_width = 4;
        marker_size = 100;
        
        plot(ratios, dll, ...
            'DisplayName', name, ...
            'LineWidth', line_width, ...
            'LineStyle', '-', ...
            'Marker', '.', ...
            'MarkerSize', marker_size);
        
        set(gca(), 'XScale', 'log');
    end
end

% all - dll - max
function plot3(thresholds)
    
    n = 100;
    info = get_info();
    
    % data
    data = get_dll_vs_ratios(thresholds);

    % [threshold, neuron, trn|val|tst, fix|sac, fold]
    [dll, ratios] = get_dll_vs_ratios_interp(data, n);
    
    % plot
    % - all neurons
    create_figure('DLL vs Ratios - All Neurons - Max');
    hold('on');

    for type_trials = [1, 3] % ["train", "validation", "test"]
        for type_times = 1:2 % ["fix", "sac", "all"]
            plot_dll_vs_ratios_for_a_neuron(...
                ratios, ...
                dll(:, :, type_trials, type_times), ...
                get_display_name(type_trials, type_times));
        end
    end

    hold('off');
    lgd = legend();
    title(lgd, 'Trials, Times');
    title('All Neurons');
    xlabel('\rho = #spikes / #parameters');
    ylabel('\DeltaLL/spk (bit/sec)');

    set_axis();

    saveas(...
        gcf, ...
        fullfile(...
            info.folders.assets, ...
            'results', ...
            ['dll-max', info.plotting.formattype]));
        
   % Local functions
    function display_name = get_display_name(type_trials, type_times)
        name_trials = {'train', 'validation', 'test'};
        name_times = {'fix', 'sac', 'all'};
        
        display_name = sprintf('%s - %s', ...
            name_trials{type_trials}, ...
            name_times{type_times});
    end
    
    function plot_dll_vs_ratios_for_a_neuron(ratios, dll, name)
        line_width = 4;
        
        plot(ratios, nanmax(dll, [], 2), ...
            'DisplayName', name, ...
            'LineWidth', line_width);
        
        set(gca(), 'XScale', 'log');
    end
end

% all - dll - mean
function plot4(thresholds)
    
    n = 100;
    info = get_info();
    
    % data
    data = get_dll_vs_ratios(thresholds);

    % [threshold, neuron, trn|val|tst, fix|sac, fold]
    [dll, ratios] = get_dll_vs_ratios_interp(data, n);
    
    % plot
    % - all neurons
    create_figure('DLL vs Ratios - All Neurons - Mean');
    hold('on');

    for type_trials = [1, 3] % ["train", "validation", "test"]
        for type_times = 1:2 % ["fix", "sac", "all"]
            plot_dll_vs_ratios_for_a_neuron(...
                ratios, ...
                dll(:, :, type_trials, type_times), ...
                get_display_name(type_trials, type_times));
        end
    end

    hold('off');
    lgd = legend();
    title(lgd, 'Trials, Times');
    title('All Neurons');
    xlabel('\rho = #spikes / #parameters');
    ylabel('\DeltaLL/spk (bit/sec)');

    set_axis();

    saveas(...
        gcf, ...
        fullfile(...
            info.folders.assets, ...
            'results', ...
            ['dll-mean', info.plotting.formattype]));
        
   % Local functions
    function display_name = get_display_name(type_trials, type_times)
        name_trials = {'train', 'validation', 'test'};
        name_times = {'fix', 'sac', 'all'};
        
        display_name = sprintf('%s - %s', ...
            name_trials{type_trials}, ...
            name_times{type_times});
    end
    
    function plot_dll_vs_ratios_for_a_neuron(ratios, dll, name)
        line_width = 4;
        
        % normalize (`max` -> `1`)
        for i = 1:size(dll, 2) % for each neuron
            dll(:, i) = dll(:, i) ./ max(dll(:, i));
        end
        
        % todo: fix:
        errorbar(ratios, mean(dll, 2), std(dll, 0, 2) / sqrt(size(dll, 2)), ...
            'DisplayName', name, ...
            'LineWidth', line_width);
        
        set(gca(), 'XScale', 'log');
    end
end

% all - tst/trn - mean
function plot5(thresholds)
    
    n = 100;
    info = get_info();
    
    % data
    data = get_dll_vs_ratios(thresholds);

    % [threshold, neuron, trn|val|tst, fix|sac, fold]
    [dll, ratios] = get_dll_vs_ratios_interp(data, n);
    
    % plot
    % - all neurons
    create_figure('DLL (tst/trn) vs Ratios - All Neurons - Mean');
    hold('on');

    for type_times = 1:2 % ["fix", "sac", "all"]
        plot_dll_vs_ratios_for_a_neuron(...
            ratios, ...
            dll(:, :, 3, type_times) ./ dll(:, :, 1, type_times), ...
            get_display_name(type_times));
    end

    hold('off');
    lgd = legend();
    title(lgd, 'Trials, Times');
    title('All Neurons');
    xlabel('\rho = #spikes / #parameters');
    ylabel('PR [test] / PR [train]');

    set_axis();

    saveas(...
        gcf, ...
        fullfile(...
            info.folders.assets, ...
            'results', ...
            ['dll-tst-trn-mean', info.plotting.formattype]));
        
   % Local functions
    function display_name = get_display_name(type_times)
        name_times = {'fix', 'sac', 'all'};
        
        display_name = sprintf('%s', ...
            name_times{type_times});
    end
    
    function plot_dll_vs_ratios_for_a_neuron(ratios, dll, name)
        line_width = 4;
        
        errorbar(ratios, nanmean(dll, 2), nanstd(dll, 0, 2) / sqrt(size(dll, 2)), ...
            'DisplayName', name, ...
            'LineWidth', line_width);
        
        set(gca(), 'XScale', 'log');
    end
end

function plot6(thresholds)
    
    marker_size = 10;
    info = get_info();
    
    % data
    data = get_dll_vs_ratios(thresholds);

    % [threshold, neuron, trn|val|tst, fix|sac, fold]
    dll = mean(data.dll, 5); % mean along dimension `fold`
    ratios = mean(data.ratios, 5); % mean along dimension `fold`
    
    % - all neurons
    create_figure('DLL vs Ratios - All Neurons');
    hold('on');

    for type_trials = [1, 3] % ["train", "validation", "test"]
        for type_times = 1:2 % ["fix", "sac", "all"]

            ratios_all = ratios(:, :, type_trials, type_times);
            dll_all = dll(:, :, type_trials, type_times);
            
            ratios_all = ratios_all(:);
            dll_all = dll_all(:);
            
            [~, idx] = sort(ratios_all);
            
            plot_dll_vs_ratios_for_a_neuron(...
                ratios_all(idx), ...
                dll_all(idx), ...
                get_display_name(type_trials, type_times), ...
                marker_size);

    %         % threshold-axis
    %         ax = gca();
    %         axes('Position', ax.Position, ...
    %             'XAxisLocation', 'top', ...
    %             'Color', 'none');
        end
    end

    hold('off');
    lgd = legend();
    title(lgd, 'Trials, Times');
    title('All Neurons');
    xlabel('\rho = #spikes / #parameters');
    ylabel('\DeltaLL/spk (bit/sec)');

    set_axis();

    saveas(...
        gcf, ...
        fullfile(...
            info.folders.assets, ...
            'results', ...
            ['dll-all', info.plotting.formattype]));
end

% Data
function data = get_dll_vs_ratios(thresholds)
    % global smm_threshold
    
    info = get_info();
    
    filename = fullfile(...
        info.folders.assets, ...
        'results', ...
        'dll.mat');
    
    if ~isfile(filename)
        ids = get_ids();
        num_thresholds = numel(thresholds);
        num_neurons = numel(ids);
        num_folds = info.num_folds;
        
        % (threshold, neuron, trn|val|tst, fix|sac, fold)
        dll = zeros(num_thresholds, num_neurons, 3, 2, num_folds); 
        ratios = zeros(num_thresholds, num_neurons, 3, 2, num_folds);
        
        for i_th = 1:num_thresholds
            smm_threshold = thresholds(i_th);
            set_global('smm_threshold', smm_threshold);
            
            for i_n = 1:num_neurons
                id = ids{i_n};

                fprintf('%d - %s\n', i_n, id);

                [session, channel, unit] = get_id_parts(id);

                dll(i_th, i_n, :, :, :) = ...
                    get_s_dll_for_one_neuron(session, channel, unit);
                
                ratios(i_th, i_n, :, :, :) = ...
                    get_ratios_for_one_neuron(session, channel, unit);
            end
        end

        save(filename, 'dll', 'ratios', 'thresholds', 'ids');
    end
    
    data = load(filename, 'dll', 'ratios', 'thresholds', 'ids');
end

function dll = get_s_dll_for_one_neuron(session, channel, unit)
    info = get_info();
    num_folds = info.num_folds;
    
    
    dll = nan(3, 2, num_folds);
    
    for type_trials = 1:3 % ["train", "validation", "test"]
        for type_times = 1:2 % ["fix", "sac", "all"]
            for fold = 1:num_folds
                try
                    dll(type_trials, type_times, fold) = get_dll(...
                        session, ...
                        channel, ...
                        unit, ...
                        fold, ...
                        type_trials, ...
                        type_times);
                catch ME
                    save_error(ME);
                end
            end
        end
    end
end

function ratios = get_ratios_for_one_neuron(session, channel, unit)
    info = get_info();
    num_folds = info.num_folds;
    
    ratios = nan(3, 2, num_folds);
    
    for type_trials = 1:3 % ["train", "validation", "test"]
        for type_times = 1:2 % ["fix", "sac", "all"]
            for fold = 1:num_folds
                try
                    % number of responses
                    resp = get_resp(session, channel, unit, fold);
                    trials = get_trials(session, channel, unit, fold, type_trials);
                    nr = sum(resp(trials, :), 'all');

                    % number of parameters
                    rbmap = get_rbmap(session, channel, unit);
                    times = get_times(type_times, true);
                    np = sum(rbmap(:, :, times, :), 'all');

                    ratios(type_trials, type_times, fold) = nr ./ np;
                catch ME
                    save_error(ME);
                end
            end
        end
    end
end

function [dll_interp, ratios_interp] = get_dll_vs_ratios_interp(data, n)

    % [threshold, neuron, trn|val|tst, fix|sac, fold]
    dll = nanmean(data.dll, 5); % mean along dimension `fold`
    ratios = nanmean(data.ratios, 5); % mean along dimension `fold`
    ids = data.ids;
    
    dll_interp = zeros(n, size(dll, 2), size(dll, 3), size(dll, 4));
    ratios_interp = linspace(nanmin(ratios(:)), nanmax(ratios(:)), n);
    xq = ratios_interp;
    
    for i = 1:numel(ids)
        for type_trials = [1, 3] % ["train", "validation", "test"]
            for type_times = 1:2 % ["fix", "sac", "all"]
                
                x = ratios(:, i, type_trials, type_times);
                y = dll(:, i, type_trials, type_times);
                
                % ignore `nan`
                idx = isnan(y);
                x(idx) = [];
                y(idx) = [];
                
                % interp
                yq = interp1(x, y, xq);
                % `nearest` extrap
                % - left
                idx = xq < x(1);
                yq(idx) = y(1);
                % - right
                idx = xq > x(end);
                yq(idx) = y(end);
                
                dll_interp(:, i, type_trials, type_times) = yq;
            end
        end
    end
    
end