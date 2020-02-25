function [dll_evolution, ids, delta] = get_s_dll_evolution()
    % Plot delta log likelihood evolution of specific s-model for given trial
    %
    % Parameters
    % ----------
    %

    delta = 75;
    
    info = get_info();
    
    filename = fullfile(...
        info.folders.results, ...
        sprintf('s_dll_evolution_delta_%d.mat', delta));
    
    if isfile(filename)
        file = load(filename);
        dll_evolution = file.dll_evolution;
        ids = file.ids;
        delta = file.delta;
        
        return
    end
    
    num_folds = info.num_folds;
    num_times = info.num_times;

    ids = get_ids();
    num_neurons = numel(ids);

    dll_evolution = nan(num_neurons, num_folds, num_times, 2); % 2 means for `test` and `train` trials

    fprintf('Make delta log likelihood evolution for the S-Model\n');
    tic();

    for i = 1:num_neurons % todo: parfor
        id = ids{i};

        fprintf('%d - %s\n', i, id);

        [session, channel, unit] = get_id_parts(id);

        dll_evolution(i, :, :, :) = get_s_dll_evolution_for_one_neuron(session, channel, unit, delta);
    end

    toc();

    save(filename, 'dll_evolution', 'ids', 'delta');
end

function dll_evolution = get_s_dll_evolution_for_one_neuron(session, channel, unit, delta)
    info = get_info();
    num_folds = info.num_folds;
    num_times = info.num_times;

    dll_evolution = nan(num_folds, num_times, 2);
    for fold = 1:num_folds
        for it = (delta + 1):(num_times - delta)
            times = (it - delta):(it + delta);
            
            % todo: fix: `times`
            dll_evolution(fold, it, 1) = get_dll(session, channel, unit, fold, 'test, validation', times);
            dll_evolution(fold, it, 2) = get_dll(session, channel, unit, fold, 'train', times);
        end
    end
end
