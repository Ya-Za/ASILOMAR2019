function main(neuron_idx)
    % Main function
    %
    % to see log file
    % >> type('log.txt');
    %
    % Notations
    % ---------
    % - trn  : train
    % - val  : validation
    % - tst  : test
    % - stm  : stimulus kernel
    % - psk  : post spike kernel
    % - off  : offset kernel
    % - idx  : indexes
    % - resp : responses
    % - stim : stimuli
    % - prb  : probe
    % - fr   : firing rate
    % - fr_ss: self sufficient firing rate
    % - sig  : signal
    % - ctl  : contrl
    %
    % Parameters
    % ----------
    % - session: scalar
    %   Session number with format: yymmdd
    % - channel: scalar
    %   Channel number
    % - unit: scalar
    %   Unit number

    % close all figures and clear command window
    close('all');
    clc();
    
    % global variables
    % global smm_threshold
    
    smm_thresholds = 0.3:0.1:0.8;
    poolsize = 3;
    
    % add `lib`, and all its subfolders to the path
    addpath(genpath('lib'));

    % copy command widnow to `log.txt` file
    % diary('log.txt');

    % display current date/time
    disp(datetime('now'));

    % start main timer
    main_timer = tic();
    
    ids = get_ids();
    if nargin >= 1
        neuron_idx = neuron_idx(neuron_idx <= numel(ids));
        ids = ids(neuron_idx);
    end
    num_neurons = numel(ids);
    
    delete(gcp('nocreate'));
    parpool(poolsize);
    
    for smm_threshold = flip(smm_thresholds)
        fprintf('\n\nThreshold: %d\n\n', smm_threshold);
        
        % change_threshold(th);
        set_global('smm_threshold', smm_threshold);
        
        parfor i = 1:num_neurons % todo: parfor    
            try
                id = ids{i};

                fprintf('%d - %s\n', i, id);

                [session, channel, unit] = get_id_parts(id);

                % S-Model
                smodel(session, channel, unit);

                % F-Model
                % fmodel(session, channel, unit);

                % A-Model
                % amodel(session, channel, unit);

                % Plot some results
                plot_results(session, channel, unit)

                close('all');
            catch ME
                save_error(ME);
            end
        end
    end

    % Performance on all neurons
    % plot_s_dll_vs_ratios(smm_thresholds);

    fprintf('\n\n');
    toc(main_timer);
    % diary('off');
end

% Functions
function change_threshold(th)
    
    get_info_filename = './lib/get/get_info.m';
    
    % read
    txt = fileread(get_info_filename);
    % replace
    txt = regexprep(txt, 'smm_threshold = \d+\.\d+;', sprintf('smm_threshold = %f;', th));
    % write
    file = fopen(get_info_filename, 'w');
    fwrite(file, txt);
    fclose(file);

    pause(10);
end
