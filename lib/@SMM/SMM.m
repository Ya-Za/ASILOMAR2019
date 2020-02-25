classdef SMM < handle
    % Spatiotemporal sensitivity map modeling
    
    properties (Constant)
        ISSURF = false;
        TESTPCT = 0.1; % proportion of test data
%         FOLDERS = struct(...
%             'ASSETS', '/uufs/chpc.utah.edu/common/home/noudoost-group1/yasin/smm/assets/41/', ...
%             'DATA', '/uufs/chpc.utah.edu/common/home/noudoost-group1/yasin/smm/assets/41/data', ...
%             'SMODELS', '/uufs/chpc.utah.edu/common/home/noudoost-group1/yasin/smm/assets/41/smodels', ...
%             'RESULTS', '/uufs/chpc.utah.edu/common/home/noudoost-group1/yasin/smm/assets/41/results-z-15', ...
%             'MODELS', '/uufs/chpc.utah.edu/common/home/noudoost-group1/yasin/smm/assets/41/results-z-15/models');

        FOLDERS = struct(...
            'ASSETS', 'D:/data/smm/assets/41/sfn19-plos', ...
            'DATA', 'D:/data/smm/assets/41/data-plos', ...
            'SMODELS', 'D:/data/smm/assets/41/smodels', ...
            'RESULTS', 'D:/data/smm/assets/41/procedure-z-window-7-mask-1-test-10', ...
            'MODELS', 'D:/data/smm/assets/41/procedure-z-window-7-mask-1-test-10/models');
        
        RBMAPSIZE = [156, 30]; % [156, 23] = floor([T, D] / 7) + 2;
    end
    
    properties
        folders % folder pathes
        filenames % input file names
        flags % flags
        
        width % width of spatial kernel
        height % height of spatial kernel
        tmin % minimum time
        tmax % maximum time
        dmin % minimum delay
        dmax % maximum delay
        window % window length of smoothing
        threshold % threshold of noise removing
        procedureName % 'z', 'x2', 'roc', 'sta'
        erodeR % radius of erosion 
        dilateR % radius of dilation
        
        models % cell array of trained logistics (nonlinear parts) for each time
        kernels % linear kernels (spatiotemporal sensitivity maps) for each time
        
        dataidx % training/test data indeces
        
        dbounds % delay bounds
    end
    
    % Constructor
    methods
        function this = SMM(filename, assets, props)
            
            if nargin < 1
                return
            end
            
            if nargin < 2
                assets = SMM.FOLDERS.ASSETS;
            end
            
            % constructor
            this.width = 9;
            this.height = 9;
            this.tmin = -540; %? -350
            this.tmax = 540;
            this.dmin = 0;
            this.dmax = 199;
            this.window = 1;
            this.threshold = 0.9;
            this.procedureName = 'z';
            this.erodeR = 0;
            this.dilateR = 0;
            this.flags = struct(...
                'mask', false, ...
                'test', 'no');
            
            if exist('props', 'var')
                this.setProps(props);
            end
            
            % folders
            this.initFolders(assets);
            
            % filenames
            this.initFilenames(filename);
            
            % dataidx
            this.initDataidx();
            
            if ~isfile(this.filenames.model)
                this.saveConfig();
            end
        end
        
        function initFolders(this, assets)
            results = SMM.getResultsFolder(...
                this.procedureName, ... 
                this.window, ...
                this.threshold, ...
                this.erodeR, ...
                this.dilateR);
            results = fullfile(assets, results);

            this.folders = struct(...
                'assets', assets, ...
                'results', results, ...
                'models', fullfile(results, 'models'));
            
            
            % make directories
            % fields = fieldnames(this.folders);
            fields = {'results', 'models'};
            for i = 1:numel(fields)
                field = fields{i};
                
                folder = this.folders.(field);
                if ~exist(folder, 'dir')
                    mkdir(folder);
                end
            end
        end
        
        function initFilenames(this, dataFilename)
            [~, name] = fileparts(dataFilename);
            modelFilename = [name, '.mat'];
            
            this.filenames = struct(...
                'data', dataFilename, ...
                'model', fullfile(this.folders.models, modelFilename));
        end
        
        function setProps(this, props)
            
            fields = fieldnames(props);
            for i = 1:numel(fields)
                field = fields{i};
                this.(field) = props.(field);
            end
        end
        
        function saveConfig(this)
            % Save configuration
            
            fprintf('\nSave configuration (`config`) in `%s`: ', this.filenames.model);
            tic();
            
            config = struct(...
                'filename', this.filenames.data, ...
                'assets', this.folders.assets, ...
                'flags', this.flags, ...
                'width', this.width, ...
                'height', this.height, ...
                'tmin', this.tmin, ...
                'tmax', this.tmax, ...
                'dmin', this.dmin, ...
                'dmax', this.dmax, ...
                'window', this.window, ...
                'threshold', this.threshold, ...
                'procedureName', this.procedureName, ...
                'erodeR', this.erodeR, ...
                'dilateR', this.dilateR, ...
                'dataidx', this.dataidx);
            
            if exist(this.filenames.model, 'file')
                save(this.filenames.model, 'config', '-append');
            else
                save(this.filenames.model, 'config');
            end
            toc();
        end
        
        function loadModels(this)
            this.models = this.getModels();
        end
        
        function loadKernels(this)
            this.kernels = this.getKernels();
        end
    end
    
    % init data indeces
    methods
        function initDataidx(this)
            
            % no test data
            % smodel
            % cross-validation
            
            switch this.flags.test
                case 'no'
                    this.initDataidxNo();
                case 'smodel'
                    this.initDataidxSModel();
                otherwise
                    this.initDataidxCV();
            end
        end
        
        function idx = getValidIdx(this)
            % load(this.filenames.data, 'set_of_trials');
            % idx = [set_of_trials.trn_indices, set_of_trials.crs_indices, set_of_trials.tst_indices];
            
            load(this.filenames.data, 'resp');
            idx = find(any(resp, 2));
        end
        
        function initDataidxNo(this)
            
            idx = this.getValidIdx();
            
            this.dataidx.test = idx;
            this.dataidx.train = idx;
        end
        
        function initDataidxSModel(this)
            
            load(this.filenames.data, 'set_of_trials');

            idx = [set_of_trials.trn_indices, set_of_trials.crs_indices, set_of_trials.tst_indices];
            
            % test
            this.dataidx.test = [];
            for i = 1:numel(set_of_trials.tst_indices)
                this.dataidx.test(end + 1) = find(idx == set_of_trials.tst_indices(i), 1);
            end
            
            % train
            this.dataidx.train = [];
            for i = 1:numel(set_of_trials.trn_indices)
                this.dataidx.train(end + 1) = find(idx == set_of_trials.trn_indices(i), 1);
            end
            
            % val
            for i = 1:numel(set_of_trials.crs_indices)
                this.dataidx.train(end + 1) = find(idx == set_of_trials.crs_indices(i), 1);
            end
        end
        
        function initDataidxCV1(this)
            % k-fold cross-validation
            
            idx = this.getValidIdx();
            cv = sscanf(this.flags.test, '%d'); % cv = [i, k] := iteration `i` of `k`
            if numel(cv) == 1
                this.dataidx.test = idx;
                this.dataidx.train = idx;
                return
            end
            
            n = numel(idx); % number of trials (all data)
            m = floor(n / cv(2)); % number of test data
            
            i1 = (cv(1) - 1) * m + 1; % beginning index
            i2 = i1 + m - 1;% ending index
            
            testIdx = false(n, 1);
            testIdx(i1:i2) = true;
            
            this.dataidx.test = idx(testIdx);
            this.dataidx.train = idx(~testIdx);
        end
        
        function initDataidxCV(this)
            % k-fold cross-validation
            
            idx.all = this.getValidIdx();
            n = numel(idx.all);
            m = floor(SMM.TESTPCT * n);
            idx.test = idx.all(1:m);
            idx.train = idx.all(m+1:end);
            
            cv = sscanf(this.flags.test, '%d'); % cv = [i, k] := iteration `i` of `k`
            if numel(cv) == 1
                this.dataidx.test = idx.test;
                this.dataidx.train = idx.train;
                return
            end
            
            n = numel(idx.train); % number of train trials
            m = floor(n / cv(2)); % number of test data for this fold
            
            i1 = (cv(1) - 1) * m + 1; % beginning index
            i2 = i1 + m - 1;% ending index
            
            testIdx = false(n, 1);
            testIdx(i1:i2) = true;
            
            this.dataidx.test = idx.train(testIdx);
            this.dataidx.train = idx.train(~testIdx);
        end
        
        function initDataidx1(this)
            load(this.filenames.data, 'stim'); % stimuli
            n = size(stim, 1); % number of trials (all data)
            m = ceil(SMM.TESTPCT * n); % number of test data
            
            this.dataidx.test = 1:m;
            this.dataidx.train = (m + 1):n;
        end
        
        function initDataidx2(this)
            load(this.filenames.data, 'cond');
            ucond = unique(double(cond)); % unique conditions
            n = numel(ucond); % number of all conditions
            m = ceil(SMM.TESTPCT * n); % number of test conditions
            
            tcond = ucond(1:m); % test conditions
            
            this.dataidx.test = [];
            this.dataidx.train = [];
            
            for i = 1:numel(cond)
                if ismember(cond(i), tcond)
                    this.dataidx.test(end + 1) = i;
                else
                    this.dataidx.train(end + 1) = i;
                end
            end
        end
        
        function initDataidx3(this)
            load(this.filenames.data, 'stim'); % stimuli
            n = size(stim, 1); % number of trials (all data)
            m = ceil(SMM.TESTPCT * n); % number of test data
            
            idx = randperm(n);
            this.dataidx.test = idx(1:m);
            this.dataidx.train = idx((m + 1):n);
        end
        
        function initDataidx4(this)
            load(this.filenames.data, 'set_of_trials');

            idx = [set_of_trials.trn_indices, set_of_trials.crs_indices, set_of_trials.tst_indices];

            if ~this.flags.test
                this.dataidx.test = idx;
                this.dataidx.train = idx;
                return
            end
            
            % test
            this.dataidx.test = [];
            for i = 1:numel(set_of_trials.tst_indices)
                this.dataidx.test(end + 1) = find(idx == set_of_trials.tst_indices(i), 1);
            end
            
            % train
            this.dataidx.train = [];
            for i = 1:numel(set_of_trials.trn_indices)
                this.dataidx.train(end + 1) = find(idx == set_of_trials.trn_indices(i), 1);
            end
            
            % val
            for i = 1:numel(set_of_trials.crs_indices)
                this.dataidx.train(end + 1) = find(idx == set_of_trials.crs_indices(i), 1);
            end
        end
    end
    
    methods (Static)
        function smm = load(filename)
            % Save configuration
            
            fprintf('\nLoad configuration (`config`) from `%s`: ', filename);
            tic();
            
            load(filename, 'config');
            
            % assets = fileparts(fileparts(fileparts(filename)));
            
            smm = SMM();
            
            smm.width = config.width;
            smm.height = config.height;
            smm.tmin = config.tmin;
            smm.tmax = config.tmax;
            smm.dmin = config.dmin;
            smm.dmax = config.dmax;
            smm.window = config.window;
            smm.threshold = config.threshold;
            smm.procedureName = config.procedureName;
            smm.erodeR = config.erodeR;
            smm.dilateR = config.dilateR;
            smm.flags = config.flags;
            smm.initFolders(config.assets);
            smm.initFilenames(config.filename);
            smm.dataidx = config.dataidx;
            
            toc();
        end
        
        function resultsFolder = getResultsFolder1(procedureName, window, mask, test)
            resultsFolder = sprintf(...
                    'procedure-%s-window-%d-mask-%d-test-%s', ...
                    procedureName, window, mask, test);
        end
        
        function resultsFolder = getResultsFolder(...
                procedureName, ...
                window, ...
                threshold, ...
                erodeR, ...
                dilateR)
            resultsFolder = sprintf(...
                    'procedure-%s-window-%d-threshold-%.2f-erodeR-%d-dilateR-%d', ...
                    procedureName, window, threshold, erodeR, dilateR);
        end
    end
    
    % Modeling (Coding)
    methods
        function fitModel(this)
            % Fit model
            
            % save config
            this.saveConfig();
            
            % require `x`, `y`
            x = this.getPredictor();
            y = this.getTrueResponse();
            
            % training data
            x = x(this.dataidx.train, :, :);
            y = y(this.dataidx.train, :);
            
            sprintf('\nFit model:\n');
            tic();
            
            times = this.getTimes(); % time of interests
            T = numel(times); % number of times
            
            this.models = cell(T, 1);
            for t = 1:T
                fprintf('Time: %d\n', times(t));
                
%                 this.models{t} = fitglm(...
%                     squeeze(x(:, :, t)), ...
%                     squeeze(y(:, t)), ...
%                     'linear', ...
%                     'Distribution', 'binomial', ...
%                     'Link', 'logit');

                [B, FitInfo] = lassoglm(...
                    squeeze(x(:, :, t)), ...
                    squeeze(y(:, t)), ...
                    'binomial', ...
                    'CV', 3);
                idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
                B0 = FitInfo.Intercept(idxLambdaMinDeviance);
                coef = [B0; B(:, idxLambdaMinDeviance)];
                this.models{t} = GLM(coef);
            end
            
            models = this.models;
            save(this.filenames.model, 'models', '-append');
            toc();
        end
        
        function resp = getDataResponse(this)
            load(this.filenames.data, 'resp');
            times = this.getTimes();
            tidx = times + 541;
            resp = resp(:, tidx);
        end
        
        function y = getTrueResponse(this)
            info = who('-file', this.filenames.model);
            if ~ismember('y', info)
                this.makeTrueResponse();
            end
            
            load(this.filenames.model, 'y');
        end
        
        function makeTrueResponse(this)
            % true response (`y`)
            fprintf('\nMake true response (`y`):\n');
            tic();
            
            [~, resp] = this.getStimuliResponses();
            y = logical(resp);
            
            y = y(:, (this.dmax + 1):end);
            
            save(this.filenames.model, 'y', '-append');
            toc();
        end
        
        function x = getPredictor(this, removedEffectName)
            
            if nargin < 2
                removedEffectName = '';
            end
            
            info = who('-file', this.filenames.model);
            
            predictorName = ['x', removedEffectName];
                
            if ~ismember(predictorName, info)
                this.makePredictor(removedEffectName);
            end

            S = load(this.filenames.model, predictorName);
            x = S.(predictorName);
        end
        
        function makePredictor1(this, removedEffectName)
            % Make predictor/response variables
            
            if nargin < 2
                removedEffectName = '';
            end
            
            % require `STIM`, `y`, `W`
            % stimuli (`STIM`), responses (`y`)
            [stim, ~] = this.getStimuliResponses();
            
            STIM = this.code2stim(stim);
            
            % weights/kernels (`W`)
            W = this.getKernels();
            if ~isempty(removedEffectName)
                W = this.removeEffect(W, removedEffectName);
            end
            
            % predictor variables (`x`)
            fprintf('\nMake predictor (`x`):\n');
            tic();
            
            % time/delay sensitivity map
            times = this.getTimes();
            delays = this.getDelays();
            
            N = size(STIM, 1); % number of trials
            T = numel(times); % number of times
            D = numel(delays); % number of delays
            
            x = zeros(N, T);
            for t = 1:T
                fprintf('Time: %d\n', times(t));
                for i = 1:N
                    x(i, t) = sum(...
                        squeeze(W(:, :, t, :)) .* ...
                        squeeze(STIM(i, :, :, (t + D - 1):-1:t)), 'all');
                end
            end
            
            S.(['x', removedEffectName]) = x;
            save(this.filenames.model, '-struct', 'S', '-append');
                
            toc();
        end
        
        function makePredictor(this, removedEffectName)
            % Make predictor/response variables
            
            if nargin < 2
                removedEffectName = '';
            end
            
            % require `STIM`, `y`, `W`
            % stimuli (`STIM`), responses (`y`)
            [stim, ~] = this.getStimuliResponses();
            
            STIM = this.code2stim(stim);
            
            % weights/kernels (`W`)
            W = this.getKernels();
            if ~isempty(removedEffectName)
                W = this.removeEffect(W, removedEffectName);
            end
            
            % predictor variables (`x`)
            fprintf('\nMake predictor (`x`):\n');
            tic();
            
            % time/delay sensitivity map
            times = this.getTimes();
            delays = this.getDelays();
            
            N = size(STIM, 1); % number of trials
            T = numel(times); % number of times
            D = numel(delays); % number of delays
            L = this.width * this.height; % number of probe locations
            sz = [this.width, this.height];
            
            x = zeros(N, L, T);
            for t = 1:T
                fprintf('Time: %d\n', times(t));
                for i = 1:N
                    for l = 1:L
                        [ix, iy] = ind2sub(sz, l);
                        x(i, l, t) = sum(...
                            squeeze(W(ix, iy, t, :)) .* ...
                            squeeze(STIM(i, ix, iy, (t + D - 1):-1:t)), 'all');
                    end
                end
            end
            
            S.(['x', removedEffectName]) = x;
            save(this.filenames.model, '-struct', 'S', '-append');
                
            toc();
        end
        
        function W = removeEffect(this, W, effectName)
            effect = this.getEffect(effectName);
            
            probe = effect.probe;
            x = probe(1);
            y = probe(2);
            
            tilim = effect.time.sac; % presaccadic time limits
            times = tilim(1):tilim(2); % presaccadic times
            tidx = times - this.tmin + 1;% to time indeces
            tnum = numel(times); % number of presaccadic times
            
            dlim = effect.time.window; % delay limits
            delays = dlim(1):dlim(2); % delays
            didx = delays - this.dmin + 1; % to delay indeces
            dnum = numel(delays); % number of delays
            
            for it = 1:tnum
                for id = 1:dnum
                    W(x, y, tidx(it) + delays(id), didx(id)) = 0;
                end
            end
            
            % W(x, y, times, :) = W(x, y, flip(times - tnum),:);
            % W(x, y, times, :) = zeros(tnum, size(W, 4));
        end
        
        function [stim, resp] = getStimuliResponses(this)
            info = who('-file', this.filenames.model);
            if ~ismember('stim', info) || ~ismember('resp', info)
                this.makeStimuliResponses();
            end
            
            load(this.filenames.model, 'stim', 'resp');
        end
        
        function makeStimuliResponses(this)
            % Load, align and smooth data

            % load
            fprintf('\nLoad stimuli/responses from `%s`: ', this.filenames.data);
            tic();
            
            file = load(this.filenames.data);
            STIM = double(file.stim);
            RESP = double(file.resp);
            if isfield(file, 'tsac')
                tsac = double(file.tsac);
            else
                tsac = ceil(size(STIM, 2) / 2) * ones(size(STIM, 1), 1);
            end
            
            toc();
            
            % align
            fprintf('Align stimuli/responses to saccade: ');
            tic();
            times = (this.tmin - this.dmax):this.tmax;
            
            N = numel(tsac); % number of trials
            T = numel(times); % number of times
            
            stim = zeros(N, T);
            resp = zeros(N, T);
            
            for i = 1:N
                t = times + tsac(i);
                
                stim(i, :) = STIM(i, t);
                resp(i, :) = RESP(i, t);
            end
            
            toc();
            
            % smooth
            fprintf('Smooth stimuli/responses: ');
            tic();
            
            resp = smoothdata(resp, 2, 'gaussian', this.window);
            
            save(this.filenames.model, 'stim', 'resp', '-append');
            toc();
        end
        
        function M = getModels(this)
            fprintf('\nGet nonlinear transforms: ');
            tic();
            
            S = load(this.filenames.model, 'models');
            M = S.models;
            
            toc();
        end
        
        function W = getKernels(this)
            
            fprintf('\nGet linear kernels: ');
            tic();
            
            info = who('-file', this.filenames.model);
            if ~ismember('W', info)
                if this.flags.mask
                    W = this.getMMap();
                else
                    W = this.getSMap();
                end
                
                % todo: must be thought
                W(isnan(W)) = 0;
                for x = 1:this.width
                    for y = 1:this.height
                        W(x, y, :, :) = imgaussfilt(squeeze(W(x, y, :, :)), 1);
                    end
                end
                
                save(this.filenames.model, 'W', '-append');
            end
            
            load(this.filenames.model, 'W');
            W(isnan(W)) = 0;
            
            toc();
        end
    end
    
    % - Helper methods
    methods
        function stim = code2stim(this, stimcode)
            % Convert coded stimuli to boolean
            %
            % Parameters
            % ----------
            % - stimcode: integer matrix(trial,time)
            %   Coded stimuli
            %
            % Returns
            % -------
            % - stim: boolean array(trial,width,height,time)
            %   Stimulus
            
            % N: Number of trials
            % T: Number of times
            [N,T] = size(stimcode); % trial x time
            
            stim = zeros(N, this.width, this.height, T);
            
            sz = [this.width, this.height];
            for trial = 1:N
                for time = 1:T
                    index = stimcode(trial,time);
                    
                    if index
                        [x,y] = ind2sub(sz,index);
                        stim(trial,x,y,time) = 1;
                    end
                end
            end
        end
        
        function neuronName = getNeuronName(this)
            [~, neuronName, ~] = fileparts(this.filenames.data);
        end
        
        function probeIndex = getProbeIndex(this, probe)
            probeIndex = sub2ind([this.width, this.height], probe(1), probe(2));
        end
        
        function effect = getEffect(this, effectName)
            neuronName = this.getNeuronName();
            model = load(fullfile('./assets/models', [neuronName '.mat']));
            effect = model.effects.(effectName);
        end
    end
    
    methods (Static)
        function fullName = getEffectFullName(shortName)
            fullName = '';
            switch shortName
                case 'ss'
                    fullName = 'Saccadic suppression';
                case 'ff'
                    fullName = 'FF-remapping';
                case 'st'
                    fullName = 'ST-remapping';
                case 'pa'
                    fullName = 'Persistent activity';
            end
        end
    end
    
    % Mapping
    methods
        function smap = getSMap(this)
            info = who('-file', this.filenames.model);
            if ~ismember('smap', info)
                this.makeSMap();
            end
            
            load(this.filenames.model, 'smap');
            
            % todo: must be commented
            times = this.getTimes();
            tidx = times - this.tmin + 1;
            
            smap = smap(:, :, tidx, :);
        end
        
        function makeSMap(this)
            % Make time/delay sensitivity map
            
            % require `stim`, `resp`
            [stim, resp] = this.getStimuliResponses();
            
            % training data
            stim = stim(this.dataidx.train, :);
            resp = resp(this.dataidx.train, :);
            
            fprintf('\nMake sensitivity map (`smap`):\n');
            tic();
            
            times = this.getTimes();
            delays = this.getDelays();
            
            tnum = numel(times);
            dnum = numel(delays);
            
            switch this.procedureName
                case 'z'
                    procedure = @SMM.ptestz;
                case 'x2'
                    procedure = @SMM.ptestx2;
                case 'roc'
                    procedure = @SMM.roc;
                case 'sta'
                    procedure = @SMM.sta;
            end
            
            % probe
            sz = [this.width, this.height];
            smap = zeros(this.width, this.height, tnum, dnum);
            for x = 1:this.width
                for y = 1:this.height
                    fprintf('Probe: (%d, %d)\n', x, y);
                    probe = sub2ind(sz, x, y);
                    
                    map = nan(tnum, dnum);
                    for it = 1:tnum % index of time
                        t = it + this.dmax;
                        
                        % pref = resp(:, t); % present
                        
                        for id = 1:dnum % index of delay
                            d = delays(id);

                            idx = stim(:, t - d) == probe;

                            pref = resp(idx, t); % present
                            npref = resp(~idx, t); % absent
                            
                            if ~isempty(pref) && ~isempty(npref)
                                s = procedure(pref, npref);

                                map(it, id) = s;
                            end
%                              data(it, id) = struct('pref', pref, 'npref', npref', 'p', p);
                        end
                    end

%                     save('data.mat', 'data');
                    smap(x, y, :, :) = map;
                end
            end
            
            save(this.filenames.model, 'smap', '-append');
            toc();
        end
        
        function bmap = getBMap(this)
            load(this.filenames.model, 'config');
            if this.threshold ~= config.threshold || ...
               this.erodeR ~= config.erodeR || ...
               this.dilateR ~= config.dilateR
                this.makeBMap();
                this.saveConfig();
            end
            
            info = who('-file', this.filenames.model);
            if ~ismember('bmap', info)
                this.makeBMap();
            end
            
            load(this.filenames.model, 'bmap');
        end
        
        function makeBMap1(this)
            % Make boolean (responsive times) map
            
            % require `smap`
            smap = this.getSMap();
            
            % boolean map (`bmap`)
            fprintf('\nMake responsive times map (`bmap`):\n');
            tic();
            
            bmap = abs(smap) >= this.threshold;
            
            % errosion/dilation structuring element
            erodeSE = strel('disk', this.erodeR);
            dilateSE = strel('disk', this.dilateR);
            for x = 1:this.width
                for y = 1:this.height
                    fprintf('Probe: (%d, %d)\n', x, y);
                    
                    bmap(x, y, :, :) = imerode(squeeze(bmap(x, y, :, :)), erodeSE);
                    bmap(x, y, :, :) = imdilate(squeeze(bmap(x, y, :, :)), dilateSE);
                end
            end
            
            % imshow(squeeze(bmap(7, 6, :, :))); % todo: must be removed
            save(this.filenames.model, 'bmap', '-append');
            toc();
        end
    
        function makeBMap2(this)
            % Make boolean (responsive times) map
            
            % require `smap`
            smap = this.getSMap();
            
            % boolean map (`bmap`)
            fprintf('\nMake responsive times map (`bmap`):\n');
            tic();
            
            bmap = false(size(smap));
            for x = 1:this.width
                for y = 1:this.height
                    bmap(x, y, :, :) = imbinarize(squeeze(smap(x, y, :, :)));
                end
            end
            
            save(this.filenames.model, 'bmap', '-append');
            toc();
        end
        
        function makeBMap(this)
            % Make boolean (responsive times) map
            
            % require `smap`
            smap = this.getSMap();
            
            % boolean map (`bmap`)
            fprintf('\nMake responsive times map (`bmap`):\n');
            tic();
            
            if this.threshold >= 0
                bmap = abs(smap) >= this.threshold;
            else
                bmap = false(size(smap));
                for x = 1:this.width
                    for y = 1:this.height
                        bmap(x, y, :, :) = imbinarize(squeeze(abs(smap(x, y, :, :))));
                    end
                end
            end
            
            % errosion/dilation structuring element
            erodeSE = strel('disk', this.erodeR);
            dilateSE = strel('disk', this.dilateR);
            for x = 1:this.width
                for y = 1:this.height
                    fprintf('Probe: (%d, %d)\n', x, y);
                    
                    bmap(x, y, :, :) = imerode(squeeze(bmap(x, y, :, :)), erodeSE);
                    bmap(x, y, :, :) = imdilate(squeeze(bmap(x, y, :, :)), dilateSE);
                end
            end
            
            save(this.filenames.model, 'bmap', '-append');
            toc();
        end
        
        function rbmap = getRBMap(this, sz)
            % Get resized binary map
            
            if isfield(this.filenames, 'mapOfResolution')
                load(this.filenames.mapOfResolution, 'map_of_resolution');
                rbmap = map_of_resolution;
                return
            end
            
            if nargin < 2
                sz = SMM.RBMAPSIZE;
            end
            
            bmap = this.getBMap();
            rbmap = zeros(this.width, this.height, sz(1), sz(2));
            for x = 1:this.width
                for y = 1:this.height
                    rbmap(x, y, :, :) = imresize(squeeze(bmap(x, y, :, :)), sz);
                end
            end
        end
        
        function mmap = getMMap(this)
            info = who('-file', this.filenames.model);
            if ~ismember('mmap', info)
                this.makeMMap();
            end
            
            load(this.filenames.model, 'mmap');
        end
        
        function makeMMap(this)
            % Make masked (sensitivity for responsive times) map
            
            % require `smap`, `bmap`
            smap = this.getSMap();
            bmap = this.getBMap();
            
            % masked map (`mmap`)
            fprintf('\nMake sensitivity for responsitve times map (`mmap`):\n');
            tic();
            
            mmap = zeros(size(smap));
            mmap(bmap) = smap(bmap);
            
            save(this.filenames.model, 'mmap', '-append');
            toc();
        end
    
        function makeDBounds(this)
            % Make masked map
            
            this.dbounds = zeros(this.width, this.height, size(this.bmap, 3), 3);
            for x = 1:this.width
                for y = 1:this.height
                    map = squeeze(this.bmap(x, y, :, :));
                    
                    T = size(map, 1); % number of times
                    bounds = nan(T, 3); % [Time, Min, Max, Duration]
                    
                    for t = 1:T
                        dfirst = find(map(t, :), 1, 'first');
                        if isempty(dfirst)
                            continue;
                        end

                        dlast = find(map(t, :), 1, 'last');
                        if isempty(dlast)
                            continue;
                        end

                        bounds(t, :) = [dfirst, dlast, dlast - dfirst];
                    end
                    
                    this.dbounds(x, y, :, :) = bounds;
                end
            end
        end
        
        function times = getTimes(this)
            times = this.tmin:this.tmax;
        end
        
        function delays = getDelays(this)
            delays = this.dmin:this.dmax;
        end
        
        function skrn = getSKrn(this)
            filename = this.getSModelFilename();
           
            % load(filename, 'skrn');
            
            % todo: must be thought
            % load(filename, 'nProfile');
            nProfile = load(filename);
            skrn = nProfile.set_of_kernels.stm.knl;
            
            times = this.getTimes();
            tidx = times + 541;
            
            skrn = skrn(:, :, tidx, 1:150);
        end
        
        function [rf1, rf2] = getRF(this)
            filename = this.getSModelFilename();
            load(filename, 'effects');

            rf1 = effects.ss.probe;
            rf2 = effects.ff.probe;
        end
        
        function filename = getSModelFilename(this)
            % todo: must be thought
            if isfield(this.filenames, 'smodel')
                filename = this.filenames.smodel;
                return
            end
            
            filename = fullfile(...
                SMM.FOLDERS.SMODELS, ...
                sprintf('%s.mat', this.getNeuronName()));
        end
    end
    
    % - Procedures
    methods (Static)
        function s = ptestz(pref, npref)
            % Two-proportion z-test, pooled for H0: p1=p2
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            x1 = sum(pref);
            x2 = sum(npref);
            n1 = numel(pref);
            n2 = numel(npref);
            
            p1 = x1 / n1;
            p2 = x2 / n2;
            
            % pooled estimate of proportion
            p = (x1 + x2) / (n1 + n2);
            z = (p1 - p2) / sqrt(p * (1 - p) * ((1 / n1) + (1 / n2)));
            
%             s = 2 * normcdf(abs(z)) - 1;
            s = normcdf(z) - normcdf(-z);
        end
    
        function s = ptestx2(pref, npref)
            % Two-proportion chi-squared test for goodness of fit
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            x1 = sum(pref);
            x2 = sum(npref);
            n1 = numel(pref);
            n2 = numel(npref);
            
            % pooled estimate of proportion
            p = (x1 + x2) / (n1 + n2);
            
            % expected counts
            x1_ = n1 * p;
            x2_ = n2 * p;
            
            observed = [x1, n1 - x1, x2, n2 - x2];
            expected = [x1_, n1 - x1_, x2_, n2 - x2_];
            
            chi2stat = sum((observed - expected) .^ 2 ./ expected);
            s = chi2cdf(chi2stat, 1);
        end
    
        function s = roc(pref, npref)
            % Receiver operating characteristic (roc)
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            th = unique([pref; npref]); % thresholds
            n = numel(th); % number of unique thresholds
            
            FPR = zeros(n, 1); % false positive rate
            TPR = zeros(n, 1); % true positive rate
            
            P = numel(pref); % condition positive
            N = numel(npref); % condition negative
            
            for i = 1:n
                TPR(i) = numel(find(pref >= th(i))) / P; % true positive rate
                FPR(i) = numel(find(npref >= th(i))) / N; % false positive rate
            end
        
            auc = -trapz(FPR, TPR); % area under curve
            
            % s = 2 * abs(auc - 0.5);
            s = auc - 0.5;
        end
    
        function s = sta(pref, npref)
            % Spike-triggered average (sta)
            %
            % Parameters
            % ----------
            % - pref: vector
            %   Preferred (positive) distribution
            % - neg: vector
            %   Nonpreferred (negative) distribution
            %
            % Returns
            % -------
            % - s: number
            %   Sensitivity index
            
            x1 = sum(pref);
            x2 = sum(npref);
            
            s = x1 / (x1 + x2);
        end
    end
    
    % Performance
    % - Mapping
    methods
        function perfs = getMappingPerfs(this, probe)
            % Get performance measures
            
            % S-Kernel
            skrn = this.getSKrn();
            skrn = abs(skrn);
            
            if isempty(probe)
                bkrn = false(size(skrn));
                for x = 1:this.width
                    for y = 1:this.height
                        bkrn(x, y, :, :) = imbinarize(squeeze(skrn(x, y, :, :)));
                    end
                end
            else
                bkrn = imbinarize(squeeze(skrn(probe(1), probe(2), :, :)));
            end
            
            bkrn = bkrn(:);
            bkrn_ = ~bkrn;
            
            
            % Sensitivity Map
            bmap = this.getBMap();
            
            if isempty(probe)
            else
                bmap = squeeze(bmap(probe(1), probe(2), :, :));
            end
            
            bmap = bmap(:);
            bmap_ = ~bmap;
            
            % condition positive (P)
            P = sum(bkrn);
            % condition negative (N)
            N = sum(bkrn_);
            % true positive (TP)
            TP = sum(bkrn & bmap);
            % false positive (FP)
            FP = sum(bkrn_ & bmap);
            % true negative (TN)
            TN = sum(bkrn_ & bmap_);
            % false negative (FN)
            FN = sum(bkrn & bmap_);
            % true positive rate (TPR)
            TPR = TP / P;
            % true negative rate (TNR)
            TNR = TN / N;
            
            perfs = struct(...
                'P', P, ...
                'N', N, ...
                'TP', TP, ...
                'FP', FP, ...
                'TN', TN, ...
                'FN', FN, ...
                'TPR', TPR, ...
                'TNR', TNR);
        end
        
        function nsu = getNumSU(this, sz)
            nsu = sum(this.getRBMap(sz), 'all');
        end
        
        function nr = getNumResp(this)
            resp = this.getDataResponse();
            nr = sum(resp, 'all');
        end
        
        function r = getRatio(this, sz)
            if nargin < 2
                sz = SMM.RBMAPSIZE;
            end
            
            nsu = this.getNumSU(sz);
            nr = this.getNumResp()';

            r = nr ./ nsu;
        end
    end
    
    % - Modeling
    methods
        function r = meanresp(this, probe, tlim, respTypeFlag, removedEffectName, alignment, onDuration, removeBase)
            % Mean response
            %
            % Parameters
            % ----------
            % - probe: [number, number]
            %   Probe location
            % - tlim: [number, number]
            %   Time limits
            % - modelFlag: boolean
            %   Responses based on `model` or `data`
            %
            % Returns
            % -------
            % - r: number[delay]
            %   Mean response
            
            if nargin < 5
                removedEffectName = '';
            end
            
            if nargin < 6
                alignment = 'stim';
            end
            
            if nargin < 7
                onDuration = 7; % 1 | 7
            end
            
            if nargin < 8
                removeBase = true;
            end
            
            tlim(1) = max(tlim(1), this.tmin);
            tlim(2) = min(tlim(2), this.tmax);
            
            times = tlim(1):tlim(2);
            tidx = times - this.tmin + 1;
            tnum = numel(tidx); % number of times
            
            % require `stim`, `resp`
            r = [];
            switch respTypeFlag
                case {'data', 'model'}
                    switch respTypeFlag
                        case 'data'
                            % resp = this.getTrueResponse(); % y
                            resp = this.getDataResponse(); % y
                        case 'model'
                            resp = this.getPrediction(removedEffectName); % y_
                    end
                    
                    load(this.filenames.model, 'stim');
                    stim = stim(:, (this.dmax + 1):end); % make size same as `resp`
                    
                    % test data
                    stim = stim(this.dataidx.test, :);
                    resp = resp(this.dataidx.test, :);

                    probeIndex = this.getProbeIndex(probe);
                    
                    for it = 1:tnum
                        tind = tidx(it); % time index
                        idx = stim(:, tind) == probeIndex;
                        if ~isempty(idx)
                            r = [r; resp(idx, (tind + this.dmin):(tind + this.dmax))];
                        end
                    end
            
                case 'single probe'
                    switch alignment
                        case 'stim'
                        case 'resp'
                            times = (tlim(1) + this.dmin):(tlim(2) + this.dmax);
                            tidx = times - this.tmin + 1;
                            tnum = numel(tidx); % number of times
                    end
                    
                    for it = 1:tnum
                        r = [r; this.responseSingleProbe(probe, times(it), alignment, onDuration, removeBase)];
                    end    
            end
            
            
            if ~isempty(r)
                r = mean(r);
            end
        end
        
        function plotMeanResp(this, effects)
            rows = 3;
            cols = 4;
            mfr = 1; % max firing rate
            
            neuronName = this.getNeuronName();
            SMM.createFigure(sprintf('Neuron: %s', neuronName));
            
            effectNames = fieldnames(effects);
            n = numel(effectNames);
            baseIndex = 0;
            
            
            % Data
            for i = 1:n
                effectName = effectNames{i};
                
                effect = effects.(effectName);
                fix = this.meanresp(effect.probe, effect.time.fix, 'data');
                sac = this.meanresp(effect.probe, effect.time.sac, 'data');
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix, sac, effect.time.window, effect.has.data);
                
                title(sprintf('%s: [%d, %d]', ...
                    SMM.getEffectFullName(effectName), ...
                    effect.time.sac(1), effect.time.sac(2)));
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('Data', 'FontSize', 24, 'FontWeight', 'bold');
            
            % S-Model
            baseIndex = baseIndex + n;
            for i = 1:n
                effectName = effectNames{i};
                
                effect = effects.(effectName);
                fix = effect.smodel.fix / mfr;
                sac = effect.smodel.sac / mfr;
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix, sac, effect.time.window, effect.has.smodel);
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('S-Model', 'FontSize', 24, 'FontWeight', 'bold');
            
            % SMM
            baseIndex = baseIndex + n;
            for i = 1:n
                effectName = effectNames{i};
                
                effect = effects.(effectName);
                fix = this.meanresp(effect.probe, effect.time.fix, 'model');
                sac = this.meanresp(effect.probe, effect.time.sac, 'model');
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix, sac, effect.time.window, true);
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('SMM', 'FontSize', 24, 'FontWeight', 'bold');
            
            subplot(rows, cols, baseIndex + 4);
            xlabel('time from stimulus onset (ms)');
            ylabel('response (unit)');
            legend(...
                {sprintf('fix: [%d, %d]', effect.time.fix(1), effect.time.fix(2))
                'sac'}, ...
                'Location', 'northeast', ...
                'NumColumns', 1);

            saveas(gcf, fullfile(this.folders.results, [neuronName, '-meanresp.png']));
            
            % Local functions
            function plotFixSac(fix, sac, window, hasEffect)
                lineWidth = 4;
                
                plot(fix, 'LineWidth', lineWidth);
                hold('on');
                if hasEffect
                    plot(sac, 'LineWidth', lineWidth);
                else
                    plot(sac, 'LineWidth', lineWidth, 'LineStyle', '-.');
                end
                
%                 yticks(0:0.5:1);

                xticks(unique([0, window, 150]));
                set(gca, 'XGrid', 'on');
                
%                 axis([0, 150, 0, 1]);
                set(gca, 'FontSize', 18);
            end
        end
        
        function plotMeanRespRemoveEffect(this, effects, removedEffectName, alignment)
            rows = 3;
            cols = 4;
            mfr = 1; % max firing rate
            
            neuronName = this.getNeuronName();
            SMM.createFigure(sprintf('Neuron: %s', neuronName));
            
            effectNames = fieldnames(effects);
            n = numel(effectNames);
            baseIndex = 0;
            
            
            % Data
            for i = 1:n
                effectName = effectNames{i};
                
                effect = effects.(effectName);
                fix = this.meanresp(effect.probe, effect.time.fix, 'data');
                sac = this.meanresp(effect.probe, effect.time.sac, 'data');
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix, sac, effect.time.window, effect.has.data);
                
                title(sprintf('%s: [%d, %d]', ...
                    SMM.getEffectFullName(effectName), ...
                    effect.time.sac(1), effect.time.sac(2)));
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('Data', 'FontSize', 24, 'FontWeight', 'bold');
            
            % SMM
            baseIndex = baseIndex + n;
            for i = 1:n
                effectName = effectNames{i};
                
                effect = effects.(effectName);
                fix = this.meanresp(effect.probe, effect.time.fix, 'single probe', '', alignment);
                sac = this.meanresp(effect.probe, effect.time.sac, 'single probe', '', alignment);
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix, sac, effect.time.window, true);
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('SMM', 'FontSize', 24, 'FontWeight', 'bold');
            
            % SMM - remove effect
            baseIndex = baseIndex + n;
            if ~isempty(removedEffectName)
                this.kernels = this.removeEffect(this.kernels, removedEffectName);
            end
            for i = 1:n
                effectName = effectNames{i};
                
                effect = effects.(effectName);
                fix = this.meanresp(effect.probe, effect.time.fix, 'single probe', '', alignment);
                sac = this.meanresp(effect.probe, effect.time.sac, 'single probe', removedEffectName, alignment);
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix, sac, effect.time.window, true);
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('SMM-', 'FontSize', 24, 'FontWeight', 'bold');
            
            subplot(rows, cols, baseIndex + 4);
            xlabel('time from stimulus onset (ms)');
            ylabel('response (unit)');
            legend(...
                {sprintf('fix: [%d, %d]', effect.time.fix(1), effect.time.fix(2))
                'sac'}, ...
                'Location', 'northeast', ...
                'NumColumns', 1);

            saveas(gcf, fullfile(this.folders.results, [neuronName, '.png']));
            
            % Local functions
            function plotFixSac(fix, sac, window, hasEffect)
                lineWidth = 4;
                
                plot(fix, 'LineWidth', lineWidth);
                hold('on');
                if hasEffect
                    plot(sac, 'LineWidth', lineWidth);
                else
                    plot(sac, 'LineWidth', lineWidth, 'LineStyle', '-.');
                end
                
%                 yticks(0:0.5:1);

                xticks(unique([0, window, 150]));
                set(gca, 'XGrid', 'on');
                
%                 axis([0, 150, 0, 1]);
                set(gca, 'FontSize', 18);
            end
        end
        
        function ll = getLL(this, trials, times, isNullModel)
            % Get loglikelihood
            
            if nargin < 4
                isNullModel = false;
            end
            
            if isempty(times)
                times = this.getTimes();
            end
            
            y = this.getTrueResponse();
            % y = this.getDataResponse();
            
            % null model
            if isNullModel
                p0 = this.getP0();
                y_ = repmat(p0, size(y));
            else
                y_ = this.getPrediction();
            end
            
            % times
            tidx = times - this.tmin + 1;
            y = y(:, tidx);
            y_ = y_(:, tidx);
            
            % train/test data
            if isempty(trials)
                idx = this.getValidIdx();
                y = y(idx, :);
                y_ = y_(idx, :);
            else
                y = y(this.dataidx.(trials), :);
                y_ = y_(this.dataidx.(trials), :);
            end
            
            
            % ll = y .* log(y_ + eps) + (1 - y) .* log(1 - y_ + eps); % Bernoulli
            ll = y .* log(y_ + eps) - y_; % Poisson
            
            % ll = (y_ .^ y) .* ((1 - y_) .^ (1 - y));
            
            % ll = sum(ll, 'all');
        end
        
        function plotLL(this, LL)
            ll = this.getLL();
            ll = -sum(ll, 2);
            
            [~, name, ~] = fileparts(this.filenames.data);
            SMM.createFigure(sprintf('Neuron: %s', name));
            
            colors = lines(2);
            
            scatter(ll, LL, 'filled');
            m = min(min(ll), min(LL));
            M = max(max(ll), max(LL));
            
            hold('on');
            line([m, M], [m, M], 'LineStyle', '--', 'Color', colors(2, :), 'LineWidth', 2);
            
            title('Negative Log-Likelihood');
            xlabel('SMM');
            ylabel('S-Model');
            SMM.setFontSize();
            
            saveas(gcf, fullfile(this.folders.results, [name, '.png']));
        end
        
        function p0 = getP0(this)
            % Mean probality of spiking
            
            y = this.getTrueResponse();
            % y = this.getDataResponse();
            
             % p0, mean over training data
            p0 = mean(y(this.dataidx.train, :), 'all');
        end
        
        function ll0 = getLL0(this, trials, times)
            % Get null loglikelihood for each trial
            
            ll0 = this.getLL(trials, times, true);
        end
        
        function lls = getLLS(this, trials, times, isNullModel)
            % Get s-model loglikelihood for each trial
            
            if nargin < 4
                isNullModel = false;
            end
            
            if isempty(times)
                times = this.getTimes();
            end
            
            load(this.filenames.data, 'resp');
            y = resp;
            
            % null model
            if isNullModel
                lambda0 = mean(y(this.dataidx.train, :), 'all');
                y_ = repmat(lambda0, size(y));
            else
                load(this.filenames.data, 'modl_data');
                y_ = modl_data / 1000;
            end
            
            % times
            tidx = times + 541;    
            y = y(:, tidx);
            y_ = y_(:, tidx);
            
            % train/test data
            if isempty(trials)
                idx = this.getValidIdx();
                y = y(idx, :);
                y_ = y_(idx, :);
            else
                y = y(this.dataidx.(trials), :);
                y_ = y_(this.dataidx.(trials), :);
            end
            
            % ll = y .* log(y_ + eps) + (1 - y) .* log(1 - y_ + eps); % Bernoulli
            lls = y .* log(y_ + eps) - y_; % Poisson
            
            % lls = sum(lls, 'all');
        end
        
        function lls0 = getLLS0(this, trials, times)
            % Get null loglikelihood for each trial
            
            lls0 = this.getLLS(trials, times, true);
        end
        
        function n = getNumOfSpikes(this, trials, times, isSModel)
            
            if nargin < 4
                isSModel = false;
            end
            
            if isempty(times)
                times = this.getTimes();
            end
            
            if isSModel
                load(this.filenames.data, 'resp');
                y = resp;
                tidx = times + 541;
            else
                y = this.getTrueResponse();
                % y = this.getDataResponse();
                tidx = times - this.tmin + 1;
            end
            
            % trial/times
            % train/test data
            if isempty(trials)
                idx = this.getValidIdx();
                y = y(idx, tidx);
            else
                y = y(this.dataidx.(trials), tidx);
            end
            
            n = sum(y, 'all');
        end
        
        function ns = getNumOfSpikesS(this, trials, times)
            ns = this.getNumOfSpikes(trials, times, true);
        end
        
        function dll = getDLL(this, trials, times)
            % Get delta log likelihood
            
            ll = this.getLL(trials, times);
            ll0 = this.getLL0(trials, times);
            n = this.getNumOfSpikes(trials, times);
            
            dll = (sum(ll, 'all') - sum(ll0, 'all')) / n;
        end
        
        function dlls = getDLLS(this, trials, times)
            % Get delta log likelihood
            
            lls = this.getLLS(trials, times);
            lls0 = this.getLLS0(trials, times);
            ns = this.getNumOfSpikesS(trials, times);
            
            dlls = (sum(lls, 'all') - sum(lls0, 'all')) / ns;
        end
    end
    
    % Plotting
    % - Maps
    methods
        function plotMap(this, probe, type)
            % Plot time delay sensitivity map
            %
            % Parameters
            % ----------
            % - isBoolean: boolean
            %   Is boolean map or not
            % - map: matrix
            %   (time x delay) sensitivity map
            
            % SMM.createFigure('Spatiotemporal sensitivity map');
            
            % todo: 
            % - good names for `type` istead of `sensitivity`, ...
            % - enum instead of string
            switch type
                case 'skrn'
                    skrn = this.getSKrn();
                    map = squeeze(skrn(probe(1), probe(2), (1 + 40):(end - 40), :)); % 1081 -> 1001
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    % caxis([-1, 1]);
                    % c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c = colorbar();
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Kernel';
                case 'sensitivity'
                    smap = this.getSMap();
                    map = squeeze(smap(probe(1), probe(2), :, :));
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    caxis([-1, 1]);
                    c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Sensitivity';
                case 'boolean'
                    bmap = this.getBMap();
                    map = squeeze(bmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(double(map'));
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    titleTxt = 'Map of Effective STUs';
                 case 'resized-boolean'
                    rbmap = this.getRBMap();
                    map = squeeze(rbmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(double(map'));
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    % c = colorbar('Limits', [0, 1], 'XTick', [0, 1]);
                    % c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Effective STUs';
                case 'masking'
                    mmap = this.getMMap();
                    map = squeeze(mmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    caxis([-1, 1]);
                    c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    % titleTxt = 'Map of Sensitivity for Responsive Times';
                    titleTxt = 'Map of Responsive Times';
                case 'dbounds' % delay bounds
                    plot(...
                        smoothdata(this.dbounds, 1, 'gaussian', this.window), ...
                        'LineWidth', 6);
                    lgd = legend({'Begin', 'End', 'Length'});
                    title(lgd, 'Delay');
                    titleTxt = 'Bounds of Temporal Kernels';
            end
            
            title(titleTxt); % todo: must be thought
%             switch type
%                 case 'skrn'
%                     title(titleTxt);
%                 otherwise
%                     title({titleTxt, this.getTitleInfo(probe)});
%             end
            
            this.setTimeAxis();
            this.setDelayAxis();
            
            SMM.setFontSize();
            axis('tight');
        end
        
        function plotMapBW(this, probe, type)
            % Plot time delay sensitivity map
            %
            % Parameters
            % ----------
            % - isBoolean: boolean
            %   Is boolean map or not
            % - map: matrix
            %   (time x delay) sensitivity map
            
            % SMM.createFigure('Spatiotemporal sensitivity map');
            
            % todo: 
            % - good names for `type` istead of `sensitivity`, ...
            % - enum instead of string
            switch type
                case 'skrn'
                    skrn = this.getSKrn();
                    map = squeeze(skrn(probe(1), probe(2), (1 + 40):(end - 40), :)); % 1081 -> 1001
                    map = abs(map);
                    
                    map = imbinarize(map);
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    caxis([0, 1]);
                    c = colorbar('Limits', [0, 1], 'XTick', [0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'S-Kernel';
                case 'sensitivity'
                    smap = this.getSMap();
                    map = squeeze(smap(probe(1), probe(2), :, :));
                    
                    map = abs(map);
                    map = imbinarize(map);
                    
                    % todo: refactor plotting
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    caxis([0, 1]);
                    c = colorbar('Limits', [0, 1], 'XTick', [0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Sensitivity';
                case 'boolean'
                    bmap = this.getBMap();
                    map = squeeze(bmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(double(map'));
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'gray');
                    titleTxt = 'Map of Responsive Times';
                case 'masking'
                    mmap = this.getMMap();
                    map = squeeze(mmap(probe(1), probe(2), :, :));
                    
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    colormap(gca, 'jet');
                    caxis([-1, 1]);
                    c = colorbar('Limits', [-1, 1], 'XTick', [-1, 0, 1]);
                    c.Label.String = 'Sensitivity (unit)';
                    titleTxt = 'Map of Sensitivity for Responsive Times';
                case 'dbounds' % delay bounds
                    plot(...
                        smoothdata(this.dbounds, 1, 'gaussian', this.window), ...
                        'LineWidth', 6);
                    lgd = legend({'Begin', 'End', 'Length'});
                    title(lgd, 'Delay');
                    titleTxt = 'Bounds of Temporal Kernels';
            end
            
            switch type
                case 'skrn'
                    title(titleTxt);
                otherwise
                    title({titleTxt, this.getTitleInfo(probe)});
            end
            
            this.setTimeAxis();
            this.setDelayAxis();
            
            SMM.setFontSize();
            axis('tight');
        end
        
        function plotMapAll(this, type)
            SMM.createFigure('Spatiotemporal sensitivity map');
            
            switch type
                case 'sensitivity'
                    maps = this.getSMap();
                case 'boolean'
                    maps = this.getBMap();
                case 'masking'
                    maps = this.getMMap();
            end
            
            for x = 1:this.width
                for y = 1:this.height
                    ax = subplot(this.width, this.height, this.getIndex(x, y));
                    
                    map = squeeze(maps(x, y, :, :));
                    if SMM.ISSURF
                        surf(map');
                        view([0,90]);
                        shading('interp');
                    else
                        imagesc(map');
                        axis('xy');
                    end
                    
                    switch type
                        case {'sensitivity', 'masking'}
                            caxis(ax, [-1, 1]);
                            colormap(ax, 'jet');
                        case 'boolean'
                            caxis(ax, [0, 1]);
                            colormap(ax, 'gray');
                        case 'dbounds'
                            plot(ax, ...
                                smoothdata(squeeze(this.dbounds(x, y, :, :)), 1, 'gaussian', this.window), ...
                                'LineWidth', 6);
                    end
                    
                    xticks(ax, []);
                    yticks(ax, []);
                    axis(ax, 'tight');
                    box(ax, 'on');
                    title(sprintf('(%d, %d)', x, y));
                end
            end
            
            % first axes
            ax = subplot(this.width, this.height, this.getIndex(1, 1));
            suptitle(this.getTitleInfo());
            this.setTimeAxis();
            this.setDelayAxis();
            
            switch type
                case {'sensitivity', 'masking'}
                    colorbar(ax, ...
                        'Location', 'east', ...
                        'Limits', [-1, 1], ...
                        'XTick', [-1, 0, 1]);
                case 'dbounds'
                    lgd = legend({'Begin', 'End', 'Length'});
                    title(lgd, 'Delay');
            end
            
            name = this.getNeuronName();
            filename = fullfile(...
                this.folders.results, ...
                sprintf('%s-%s.png', name, type));
            saveas(gcf, filename);
        end
        
        function playMapAll(this, type)
            switch type
                case 'sensitivity'
                    smap = this.getSMap();
                    I = max(smap, [], 4); % todo: how to figure out minimum values?!
                case 'boolean'
                    bmap = this.getBMap();
                    I = max(bmap, [], 4);
                    % plot `Number of sources`
                    C = squeeze(sum(I, [1, 2]));
                    SMM.createFigure('Number of sources');
                    
                    % bar(C);
                    stairs(C, 'LineWidth', 4);

                    title(this.getTitleInfo());
                    
                    this.setTimeAxis();
                    
                    ylabel('Number of sources');
                    yticks(1:max(C));
                    
                    SMM.setFontSize();
                    
                    axis('tight');
                case 'masking'
                    mmap = this.getMMap();
                    I = max(mmap, [], 4);
            end
            
            I = permute(I, [2, 1, 3]);
            I = flipud(I);
            implay(I);
        end
        
        function index = getIndex(this, x, y)
            r = this.height - y + 1;
            c = x;
            index = (r - 1) * this.width + c;
        end
    end
    
    % - Model
    methods
        function plotResults(this)
            SMM.createFigure('Model parameters');
            
            rows = 3;
            cols = 2;
            lineWidth = 4;
            
            % coefficents
            [b0, b1] = this.getCoefficients();
            % - intercept
            subplot(rows, cols, 1);
            plot(b0, 'LineWidth', lineWidth);
            set(gca, 'YScale', 'log')
            this.setTimeAxis();
            title('Intercept');
            axis('tight');
            % - slope
            subplot(rows, cols, 2);
            plot(b1, 'LineWidth', lineWidth);
            set(gca, 'YScale', 'log')
            this.setTimeAxis();
            title('Slope');
            axis('tight');
            
            % R squared/p-value
            [R2, pvalue] = this.getRSquared();
            % - R2
            subplot(rows, cols, 3);
            plot(R2, 'LineWidth', lineWidth);
            set(gca, 'YScale', 'log')
            this.setTimeAxis();
            title('R^2');
            axis('tight');
            % - p-value
            subplot(rows, cols, 4);
            plot(pvalue, 'LineWidth', lineWidth);
            set(gca, 'YScale', 'log')
            this.setTimeAxis();
            title('p-value of R^2');
            axis('tight');
            
            % actual/predicted responses
            % - actual
            load(this.filenames.model, 'y');
            subplot(rows, cols, 5);
            imagesc(y);
            colormap(gca, 'gray');
            caxis([0, 1]);
            c = colorbar('Limits', [0, 1], 'XTick', [0, 1]);
            c.Label.String = 'Spike occurance (true/false)';
            title('Actual responses');
            this.setTimeAxis();
            ylabel('Trails');
            axis('tight');
            % - predicted
            y_ = this.getPrediction();
            subplot(rows, cols, 6);
            imagesc(y_);
            colormap(gca, 'gray');
            caxis([0, 1]);
            c = colorbar('Limits', [0, 1], 'XTick', [0, 1]);
            c.Label.String = 'Spike occurance (probability)';
            title('Predicted responses');
            this.setTimeAxis();
            ylabel('Trails');
            axis('tight');
            
            suptitle(this.getTitleInfo());
        end
        
        function plotTrueVsPredictedResp1(this)
            y = this.getTrueResponse();
            y_ = this.getPrediction();
            
            % test data
            y = y(this.dataidx.test, :);
            y_ = y_(this.dataidx.test, :);
            
            rows = 3;
            cols = 1;
            
            neuronName = this.getNeuronName();
            SMM.createFigure(neuronName);
            
            % True
            subplot(rows, cols, 1);
            plotResp(y);
            title('True Responses');
            
            % Prediction
            subplot(rows, cols, 2);
            plotResp(y_);
            title('Predicted Probability of Spiks');
            
            % Prediction
            subplot(rows, cols, 3);
            plotResp(imbinarize(y_));
            title('Binarize prediction by thresholding');
            
            saveas(gcf, fullfile(this.folders.results, [neuronName, '-resp.png']));
            
            % Local functions
            function plotResp(y)
                imagesc(y);
                
                % surf(double(y));
                % view([0, 90]);
                % shading('interp');
                
                xticks([1, size(y, 2)]);
                yticks([1, size(y, 1)]);
                
                xlabel('Time');
                ylabel('Trial');
                
                colormap(gca, 'gray');
                c = colorbar();
                c.Limits = [0, 1];
                c.Ticks = [0, 1];
                
                SMM.setFontSize();
            end
        end
        
        function plotTrueVsPredictedResp(this)
            % all trials
            y.all = this.getTrueResponse();
            y_.all = this.getPrediction();
            
            % train trials
            y.train = y.all(this.dataidx.train, :);
            y_.train = y_.all(this.dataidx.train, :);
            
            % test trials
            y.test = y.all(this.dataidx.test, :);
            y_.test = y_.all(this.dataidx.test, :);
            
            rows = 2;
            cols = 2;
            
            neuronName = this.getNeuronName();
            SMM.createFigure(neuronName);
            
            %? Train
            % - True
            subplot(rows, cols, 1);
            plotResp(y.train);
            %? colorbar('off');
            title('True Responses');
            ylabel(sprintf('Train\nTrial'));
            % - Prediction
            subplot(rows, cols, 2);
            plotResp(y_.train);
            title('Predicted Probability of Spiks');
            
            %? Test
            % - True
            subplot(rows, cols, 3);
            plotResp(y.test);
            %? colorbar('off');
            title('True Responses');
            ylabel(sprintf('Test\nTrial'));
            % - Prediction
            subplot(rows, cols, 4);
            plotResp(y_.test);
            title('Predicted Probability of Spiks');
            
            folder = fullfile(this.folders.results, 'train-test-resp');
            if ~exist(folder, 'dir')
                mkdir(folder);
            end
            saveas(gcf, fullfile(folder, [neuronName, '-resp.png']));
            
            % Local functions
            function plotResp(y)
                imagesc(y);
                
                % surf(double(y));
                % view([0, 90]);
                % shading('interp');
                
                xticks([1, size(y, 2)]);
                yticks([1, size(y, 1)]);
                
                xlabel('Time');
                ylabel('Trial');
                
                colormap(gca, 'gray');
                c = colorbar();
                c.Limits = [0, 1];
                c.Ticks = [0, 1];
                
                SMM.setFontSize();
            end
        end
    end
    
    methods (Static)
        function plotTrueVsPredictedRespAllNeurons(folder)
            listing = dir(fullfile(SMM.FOLDERS.ASSETS, folder, 'models/*.mat'));
            
            fprintf('Plot true vs. predicted Responses:\n');
            tic();
            parfor i = 1:numel(listing)
                filename = fullfile(listing(i).folder, listing(i).name);
                fprintf('%d - %s\n', i, filename);
                
                smm = SMM.load(filename);
                smm.plotTrueVsPredictedResp();
                
                close('all');
            end
            toc();
        end
    end
    
    % - Helper methods
    methods
        function [b0, b1] = getCoefficients(this)
            info = who('-file', this.filenames.model);
            if ~ismember('b0', info) || ~ismember('b1', info)
                this.makeCoefficients();
            end
            
            load(this.filenames.model, 'b0', 'b1');
        end
        
        function makeCoefficients(this)
            % load
            this.loadModels();
            
            % coefficients
            T = numel(this.models); % number of times
            b0 = zeros(T, 1); % intercepts
            b1 = zeros(T, 1); % slopes
            
            for t = 1:T
                b0(t) = this.models{t}.Coefficients.Estimate(1);
                b1(t) = this.models{t}.Coefficients.Estimate(2);
            end
            
            % save
            save(this.filenames.model, 'b0', 'b1', '-append');
        end
        
        function [R2, pvalue] = getRSquared(this)
            info = who('-file', this.filenames.model);
            if ~ismember('R2', info) || ~ismember('pvalue', info)
                this.makeRSquared();
            end
            
            load(this.filenames.model, 'R2', 'pvalue');
        end
        
        function makeRSquared(this)
            % load
            load(this.filenames.model, 'models');
            
            % coefficients
            T = numel(this.models); % number of times
            R2 = zeros(T, 1); % R squared
            pvalue = zeros(T, 1); % p-values of R2
            
            for t = 1:T
                R2(t) = this.models{t}.Rsquared.Ordinary;
                pvalue(t) = this.models{t}.devianceTest().pValue(2);
            end
            
            % save
            save(this.filenames.model, 'R2', 'pvalue', '-append');
        end
        
        function y_ = getPrediction(this, removedEffectName)

            if nargin < 2
                removedEffectName = '';
            end

            info = who('-file', this.filenames.model);
            
            predictionName = ['y_', removedEffectName];
                
            if ~ismember(predictionName, info)
                this.makePrediction(removedEffectName);
            end

            S = load(this.filenames.model, predictionName);
            y_ = S.(predictionName);
            
            % todo: must be thought
            p0 = this.getP0();
            y_(y_ < p0) = p0;
        end
        
        function makePrediction(this, removedEffectName)
            % load
            
            if nargin < 2
                removedEffectName = '';
            end
            
            x = this.getPredictor(removedEffectName);
            
            this.loadModels();
            
            % coefficients
            T = numel(this.models); % number of times
            y_ = zeros(size(x, 1), size(x, 3)); % predicted reponses (probability of spike)
            
            for t = 1:T
                y_(:, t) = this.models{t}.predict(squeeze(x(:, :, t)));
            end
            
            % save
            predictionName = ['y_', removedEffectName];
            S.(predictionName) = y_;
            save(this.filenames.model, '-struct', 'S', '-append');
        end
    end
    
    methods
        function titleInfo = getTitleInfo(this, probe)
            % Neuron id
            [~, neuronTxt, ~] = fileparts(this.filenames.data);
            
            % Probe location
            if exist('probe', 'var')
                probeTxt = sprintf('(%d, %d)', probe(1), probe(2));
            else
                probeTxt = 'All';
            end
            
            % Procedure name
            switch this.procedureName
                case 'z'
                    procedureTxt = 'Pooled z-test';
                case 'x2'
                    procedureTxt = 'Chi-squared test';
                case 'roc'
                    procedureTxt = 'ROC';
                case 'sta'
                    procedureTxt = 'STA';
            end
            
            % Title info
            titleInfo = sprintf(...
                'Neuron: ''%s'', Probe: %s, Procedure: ''%s''', ...
                neuronTxt, probeTxt, procedureTxt);
        end
        
        function setTimeAxis(this)
            times = this.getTimes();
            
            xlabel('Time from saccade onset (ms)');
            
            T = numel(times);
            tidx = [1, ceil(T / 2), T];
            xticks(tidx);
            
            xticklabels(string(times(tidx)));
        end
        
        function setDelayAxis(this)
            delays = this.getDelays();
            ylabel('Delay (ms)');
            % yticks(delays(1):50:delays(end));
            yticks([1, 200]);
            yticklabels({'0', '200'});
            
        end
    end
    
    methods (Static)
        function h = createFigure(name)
            % Create `full screen` figure
            %
            % Parameters
            % ----------
            % - name: string
            %   Name of figure
            %
            % Return
            % - h: matlab.ui.Figure
            %   Handle of created figure
            
            h = figure(...
                'Name', name, ...
                'Color', 'white', ...
                'NumberTitle', 'off', ...
                'Units', 'normalized', ...
                'OuterPosition', [0, 0, 1, 1] ...
            );
        end
        
        function setFontSize()
            set(gca(), 'FontSize', 24);
        end
    end
    
    % Main
    methods (Static)
        function main()
            close('all');
            clc();
            
            % copy command widnow to `log.txt` file
            diary('log.txt');
            
            fprintf('Sensitivity Map Modeling (SMM): %s\n', datetime());
            mainTimer = tic();
            
            smm = SMM();
            
%             % fit model
%             smm.fitModel();
            
            % load config
            assets = '/uufs/chpc.utah.edu/common/home/noudoost-group1/yasin/smm/assets/';
            results = 'results-z-15';
            name = '2015051115-z-15.mat';
            filename = fullfile(assets, results, name);
            
            smm.loadConfig(filename);
            
%             % decode
%             s = smm.decode(1, 200, 50);
%             
%             imagesc(s);
%             axis('xy');
%             caxis([0, 1]);
%             colorbar();
            
            % results
%             probe = [7, 6];
%             smm.plotMap(probe, 'sensitivity');
%             smm.plotMap(probe, 'boolean');
%             smm.plotMap(probe, 'masking');

%             smm.plotMapAll('sensitivity');
%             smm.plotMapAll('boolean');
%             smm.plotMapAll('masking');

%             smm.playMapAll('sensitivity'); % press `c` and set range to [0.95, 1]
%             smm.playMapAll('boolean');
%             smm.playMapAll('masking');
            
%             smm.plotResults();

            smm.plotMeanresp();

%             smm.plotLL();

            toc(mainTimer);
            
            diary('off');
            % to see log file
            % >>> type('log.txt');
        end
        
        function fit(props)
            close('all');
            clc();
            
            if nargin < 1
                props.flags.mask = false;
                props.flags.test = 'no';
            end
            
            listing = dir(fullfile(SMM.FOLDERS.DATA, '*.mat'));
            
            fprintf('\nFit Models:\n');
            mainTimer = tic();
            for i = 1:inf:numel(listing) % todo: `parfor`
                filename = fullfile(listing(i).folder, listing(i).name);
                
                [~, name] = fileparts(filename);
                fprintf('%d - %s\n', i, name);
                localTimer = tic();
                
                smm = SMM(filename, SMM.FOLDERS.ASSETS, props);
                smm.fitModel();
                smm.makePrediction();
                
                toc(localTimer);
            end
            toc(mainTimer);
        end
        
        function perfs = makeBMaps(props)
            
            if nargin < 1
                props.flags.mask = false;
                props.flags.test = 'no';
            end
            
            listing = dir(fullfile(SMM.FOLDERS.DATA, '*.mat'));
            
            fprintf('\nMake binary maps:\n');
            mainTimer = tic();
            parfor i = 1:numel(listing) % todo: `parfor`
                filename = fullfile(listing(i).folder, listing(i).name);
                
                [~, name] = fileparts(filename);
                fprintf('%d - %s\n', i, name);
                localTimer = tic();
                
                smm = SMM(filename, SMM.FOLDERS.ASSETS, props);
                smm.saveConfig();
                smm.makeBMap();
                
                % perf
                [rf1, rf2] = smm.getRF();
                perfs_rf1(i) = smm.getMappingPerfs(rf1);
                perfs_rf2(i) = smm.getMappingPerfs(rf2);
                perfs_all(i) = smm.getMappingPerfs([]);
                
                toc(localTimer);
            end
            
            % perf
            perfs = struct(...
                'props', props, ...
                'rf1', perfs_rf1, ...
                'rf2', perfs_rf2, ...
                'all', perfs_all);
            
            toc(mainTimer);
        end
        
        function makeRBMaps(props)
            rbmapFolder = fullfile(SMM.FOLDERS.ASSETS, 'rbmap');
            mkdir(rbmapFolder);
            
            listing = dir(fullfile(SMM.FOLDERS.DATA, '*.mat'));
            
            fprintf('\nMake resized binary maps:\n');
            mainTimer = tic();
            parfor i = 1:numel(listing) % todo: `parfor`
                filename = fullfile(listing(i).folder, listing(i).name);
                
                [~, name] = fileparts(filename);
                fprintf('%d - %s\n', i, name);
                localTimer = tic();
                
                smm = SMM(filename, SMM.FOLDERS.ASSETS, props);
                smm.saveConfig();
                rbmap = smm.getRBMap(SMM.RBMAPSIZE);
                
                rbmapFilename = fullfile(rbmapFolder, name);
                SMM.saveToFile(rbmapFilename, 'rbmap', rbmap);
                
                toc(localTimer);
            end
            toc(mainTimer);
        end
        
        function saveToFile(filename, field, value)
           S.(field) = value;
           save(filename, '-struct', 'S');
        end
        
        function correctFilenames()
            listing = dir('./assets/data/*.mat');
            
            mainTimer = tic();
            for i = 1:numel(listing)
                smm = SMM(fullfile(listing(i).folder, listing(i).name));
                smm.saveConfig();
            end
            toc(mainTimer);
        end
        
        function plotLoglikeAll()
            listing = dir('./assets/data/*.mat');
            
            mainTimer = tic();
            parfor i = 1:numel(listing)
                smm = SMM(fullfile(listing(i).folder, listing(i).name));
                model = load(fullfile('./assets/models', listing(i).name));
                ll = model.ll;
                
                idx = [ll.smodel.set_of_trials.trn_indices, ll.smodel.set_of_trials.tst_indices, ll.smodel.set_of_trials.crs_indices];
                sort(idx);
                LL = ll.smodel.LL(idx);
                LL = -LL;
                
                smm.plotLL(LL);
            end
            toc(mainTimer);
        end
    end
    
    % Mean response
    methods (Static)
        function plotMeanRespAll(removedEffectName, alignment)
            % Plot mean responses for all sample neurons
            
            if nargin < 1
                removedEffectName = '';
            end
            
            if nargin < 2
                alignment = 'stim';
            end
            
            close('all');
            clc();
            
            listing = dir(fullfile(SMM.FOLDERS.MODELS, '*.mat'));
            
            mainTimer = tic();
            for i = 1:numel(listing)
                fprintf('%d - Neuron: %s\n', i, listing(i).name);
                tic();
                
                model = load(fullfile(SMM.FOLDERS.SMODELS, listing(i).name));
                
                smm = SMM.load(fullfile(listing(i).folder, listing(i).name));
                smm.loadKernels(); % todo: if isempty(this.W) this.loadKernels()
                smm.loadModels();
                    
                if isempty(removedEffectName)
                    smm.plotMeanResp(model.effects);
                else
                    smm.plotMeanRespRemoveEffect(model.effects, removedEffectName, alignment);
                end
                
                close('all');
                toc();
            end
            toc(mainTimer);
        end
        
        function saveMeanRespPopulation()
            % Plot average mean response on population
            
            mfr = 1; % maximum firing rate
            
            listing = dir(fullfile(SMM.FOLDERS.MODELS, '*.mat'));
            
            mainTimer = tic();
            
            % Init `fix`, `sac` structs
            for modelName = ["data", "smodel", "smm"]
                for effectName = ["ss", "pa", "ff", "st"]
                    fix.(modelName).(effectName) = [];
                    sac.(modelName).(effectName) = [];
                end
            end
            
            % Neurons
            for neuronInd = 1:numel(listing)
                fprintf('%d - Neuron: %s\n', neuronInd, listing(neuronInd).name);
                
                model = load(fullfile(SMM.FOLDERS.SMODELS, listing(neuronInd).name));
                effects = model.effects;
                
                smm = SMM.load(fullfile(listing(neuronInd).folder, listing(neuronInd).name));
                smm.loadKernels(); % todo: if isempty(this.W) this.loadKernels()
                smm.loadModels();
                    
                
                effectNames = fieldnames(effects);
                n = numel(effectNames);

                % Effects
                for effectInd = 1:n
                    effectName = effectNames{effectInd};

                    effect = effects.(effectName);
                    
                    if ~effect.has.data
                        continue
                    end
                    
                    % Data
                    fix.data.(effectName)(end + 1, :) = smm.meanresp(effect.probe, effect.time.fix, 'data');
                    sac.data.(effectName) = [sac.data.(effectName); smm.meanresp(effect.probe, effect.time.sac, 'data')];

                    % S-Model
                    fix.smodel.(effectName) = [fix.smodel.(effectName); effect.smodel.fix / mfr];
                    sac.smodel.(effectName) = [sac.smodel.(effectName); effect.smodel.sac / mfr];
                    
                    % SMM
                    fix.smm.(effectName) = [fix.smm.(effectName); smm.meanresp(effect.probe, effect.time.fix, 'model')];
                    sac.smm.(effectName) = [sac.smm.(effectName); smm.meanresp(effect.probe, effect.time.sac, 'model')];
                end
            end
            
            % Save
            save('meanresp.mat', 'fix', 'sac');
            
            toc(mainTimer);
        end
        
        function plotMeanRespPopulation()
            rows = 3;
            cols = 4;
            
            if ~exist('meanresp.mat', 'file')
                SMM.saveMeanRespPopulation();
            end
            
            load('meanresp.mat', 'fix', 'sac');
            effects = load('effects');
            
            SMM.createFigure('Mean Response - Population');
            
            effectNames = fieldnames(effects);
            n = numel(effectNames);
            baseIndex = 0;
            
            % Data
            for i = 1:n
                effectName = effectNames{i};
                effect = effects.(effectName);
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix.data.(effectName), sac.data.(effectName), effect.time.window);
                
                title(sprintf('%s: [%d, %d]', ...
                    SMM.getEffectFullName(effectName), ...
                    effect.time.sac(1), effect.time.sac(2)));
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('Data', 'FontSize', 24, 'FontWeight', 'bold');
            
            % S-Model
            baseIndex = baseIndex + n;
            for i = 1:n
                effectName = effectNames{i};
                effect = effects.(effectName);
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix.smodel.(effectName), sac.smodel.(effectName), effect.time.window);
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('S-Model', 'FontSize', 24, 'FontWeight', 'bold');
            
            % SMM
            baseIndex = baseIndex + n;
            for i = 1:n
                effectName = effectNames{i};
                effect = effects.(effectName);
                
                subplot(rows, cols, baseIndex + i);
                plotFixSac(fix.smm.(effectName), sac.smm.(effectName), effect.time.window);
            end
            
            subplot(rows, cols, baseIndex + 1);
            ylabel('SMM', 'FontSize', 24, 'FontWeight', 'bold');
            
            subplot(rows, cols, baseIndex + 4);
            xlabel('time from stimulus onset (ms)');
            ylabel('response (unit)');
            legend(...
                {sprintf('fix: [%d, %d]', effect.time.fix(1), effect.time.fix(2))
                'sac'}, ...
                'Location', 'northeast', ...
                'NumColumns', 1);

            saveas(gcf, fullfile(SMM.FOLDERS.RESULTS, 'meanresp-population.png'));
            
            % Local functions
            function plotFixSac(fix, sac, window)
                fixse = std(fix) / sqrt(size(fix, 2)); % standard error
                fix = mean(fix);
                
                sacse = std(sac) / sqrt(size(sac, 2)); % standard error
                sac = mean(sac);
                
                lineWidth = 4;
                color = lines(2);
                
                SMM.plotShadedErrorBar(1:numel(fix), fix, fixse, color(1, :), lineWidth);
                hold('on');
                SMM.plotShadedErrorBar(1:numel(sac), sac, sacse, color(2, :), lineWidth);
                hold('off');
                
                box('on');
                
%                 yticks(0:0.5:1);

                xticks(unique([0, window, 150]));
                set(gca, 'XGrid', 'on');
                
%                 axis([0, 150, 0, 1]);
                % set(gca, 'FontSize', 11);
            end
        end
        
        function plotShadedErrorBar(x, y, e, color, lineWidth)
            h = patch(...
                'XData', [x, flip(x)], ...
                'YData', [y + e, flip(y - e)], ...
                'EdgeColor', 'none', ...
                'FaceColor', color, ...
                'FaceAlpha', 0.1);
            
            h.Annotation.LegendInformation.IconDisplayStyle = 'off';

            hold('on');
            plot(x, y, 'Color', color, 'LineWidth', lineWidth);
            hold('off');
        end
        
        function missedEffect(name)
            % name: 'ss' | 'ff' | 'st' | 'pa'
            
            close('all');
            
            effectName = SMM.getEffectFullName(name);
            fprintf('\nEffect: %s\n', effectName);
            fprintf('\nType II error\n');
            
            listing = dir('./assets/models/*.mat');
            
            N = numel(listing); % total population
            FN = 0; % false negative
            P = 0; % condition positive
            
            for i = 1:N
                model = load(fullfile(listing(i).folder, listing(i).name));
                [~, filename, ~] = fileparts(listing(i).name);
                
                if model.effects.(name).has.data % model.(name).effect.data 
                    ds = '+'; % data sign
                    P = P + 1;
                else
                    ds = '-';
                end
                
                if model.effects.(name).has.smodel % model.(name).effect.smodel
                    ms = '+'; % model sign
                else
                    ms = '-';
                end
                
                fprintf('%2d - Nuron: %s, Data: %s, Model: %s', i, filename, ds, ms);
                if ds == '+' && ms == '-'
                    FN = FN + 1;
                    
                    SMM.createFigure(sprintf('Neuron: %s, S-Model missed: %s', filename, effectName));
                    imshow(fullfile('./assets/results/mean-response', [filename, '.png']));
                    
                    fprintf('\t***');
                end
                fprintf('\n');
            end
            
            fprintf('\nTotal population: \t%g\n', N);
            fprintf('Condition positive: \t%g (%g)\n', P, P / N);
            fprintf('False negative: \t%g (%g)\n', FN, FN / P);
        end
    end
    
    methods (Static)
        function testMeanResp()
            close('all');
            clc();
            
            tic();
            
            neuron = '2015051115';
            effectName = 'pa';
            alignment = 'stim';
            onDuration = 1;
            removeBase = true;
            
            model = load(fullfile('./assets/models', [neuron, '.mat']));
            effect = model.effects.(effectName);
            window = effect.time.window;
            hasEffect = effect.has.data;
            
            smm = SMM(fullfile('./assets/data/', [neuron, '.mat']));
            smm.loadKernels();
            smm.loadModels();
            
            fix = smm.meanresp(effect.probe, effect.time.fix, 'single probe', effectName, alignment, onDuration, removeBase);
            sac = smm.meanresp(effect.probe, effect.time.sac, 'single probe', effectName, alignment, onDuration, removeBase);
            
            smm.kernels = smm.removeEffect(smm.kernels, effectName);
            sac_ = smm.meanresp(effect.probe, effect.time.sac, 'single probe', effectName, alignment, onDuration, removeBase);
            
            smm.kernels = zeros(size(smm.kernels));
            base = smm.meanresp(effect.probe, effect.time.sac, 'single probe', '', 'stim', 1, false);
            
            % plot mean responses
            lineWidth = 4;
            if hasEffect
                lineStyle = '-';
            else
                lineStyle = '-.';
            end
            
            SMM.createFigure('Test - Effect Removing');
            times = smm.dmin:smm.dmax;
            plot(times, fix, 'LineWidth', lineWidth);
            hold('on');
            plot(times, sac, 'LineWidth', lineWidth, 'LineStyle', lineStyle);
            plot(times, sac_, 'LineWidth', lineWidth, 'LineStyle', lineStyle);
            plot(times, base, 'LineWidth', lineWidth, 'LineStyle', '--');
            
            legend({'fix', 'sac', 'sac-', 'base'});

            xticks(unique([0, window, 150]));
            set(gca, 'XGrid', 'on');
            
            yticks(0:0.1:5);

            % axis([0, 150, 0, 0.5]);
            axis('tight');
            SMM.setFontSize();
            
%             % plot baselines
%             times = smm.tmin:smm.tmax;
%             tidx = times - smm.dmin + 1;
%             tnum = numel(tidx);
%             
%             b = zeros(tnum, 1);
%             for it = 1:tnum
%                 b(it) = smm.models{it}.predict(0);
%             end
%             
%             SMM.createFigure('Baseline');
%             plot(times, b, 'LineWidth', lineWidth);
%             
%             title('Baseline');
%             xlabel('Time from saccade onset (ms)');
%             ylabel('Probability of spike in the absence of stimulus');
%             
%             xticks(unique(sort([-500, 0, 500, effect.time.sac, effect.time.sac(end) + 150])));
%             set(gca, 'XGrid', 'on');
%             
%             yticks(0:0.1:5);
%             
%             axis([-500, 500, 0, 0.5]);
%             set(gca, 'FontSize', 18);
            
            
%             % Neda
%             createFigure('Neda');
%             % sac - base
%             plot(sac - base, 'LineWidth', lineWidth, 'LineStyle', lineStyle);
%             hold('on');
%             % sac(wo effect) - base
%             plot(sac_ - base, 'LineWidth', lineWidth, 'LineStyle', lineStyle);
%             
%             legend({'sac - base', 'sac- - base'});
            
            toc();
        end
    end
    
    % Make AMM
    methods (Static)
        function fitFolds(procedureName, window, mask, k)
            for i = 1:k % over folds
                test = sprintf('%d %d', i, k);
                props = struct(...
                    'procedureName', procedureName, ...
                    'window', window, ...
                    'flags', struct('mask', mask, 'test', test));

                % fit
                SMM.fit(props);
                % results
                sfn19(SMM.getResultsFolder(procedureName, window, mask, test));
            end
        end
        
        function makeAMM(procedureName, window, mask, k)
            % - k: number
            %   Number of foldes

            % AMM folder
            ammFolder = SMM.getAMMFolder(procedureName, window, mask, k);

            if exist(ammFolder, 'dir')
                return
            else
                mkdir(ammFolder);
            end

            % Neurons
            listing = dir(fullfile(SMM.getFoldFolder(1, procedureName, window, mask, k), '*.mat'));
            T = numel(listing);

            neurons = cell(T, 1);

            for i = 1:T
                neurons{i} = listing(i).name;
            end

            % Make AMM
            fprintf('Make average model:\n');
            tic();
            parfor i = 1:numel(neurons) % todo: parfor
                neuron = neurons{i};
                fprintf('\n%d. %s\n', i, neuron);

                SMM.makeAMMOneNeuron(neuron, procedureName, window, mask, k);
            end
        end
        
        function makeAMMOneNeuron(neuron, procedureName, window, mask, k)
            ammFolder = SMM.getAMMFolder(procedureName, window, mask, k);
            ammFilename = fullfile(ammFolder, neuron);
            
            % AMM
            % - config
            config = getField(1, 'config');
            config.flags.test = num2str(k);
            config.dataidx.train = unique([config.dataidx.test, config.dataidx.train]);
            
            load(config.filename, 'set_of_trials');
            idx = [set_of_trials.trn_indices, set_of_trials.crs_indices, set_of_trials.tst_indices];
            n = numel(idx);
            m = floor(SMM.TESTPCT * n);
            config.dataidx.test = idx(1:m);
            % - stim
            stim = getField(1, 'stim');
            % - resp
            resp = getField(1, 'resp');
            % - y
            y = getField(1, 'y');
%                 % - y_
%                 y_ = getField(1, 'y_');
% 
%                 for iter = 2:k
%                     y_ = y_ + getField(iter, 'y_');
%                 end
%                 y_ = y_ / k;

            % - W
            W = getField(1, 'W');

            for iter = 2:k
                W = W + getField(iter, 'W');
            end
            W = W / k;

            % - models
            M = getField(1, 'models');
            T = numel(M);
            models = cell(T, 1);
            for t = 1:T
                %? models{t} = GLM(M{t}.Coefficients{:, 1});
                models{t} = GLM(M{t}.Coeff);
            end
            for iter = 2:k
                M = getField(iter, 'models');
                for t = 1:T
                    %? models{t}.Coeff = models{t}.Coeff + M{t}.Coefficients{:, 1};
                    models{t}.Coeff = models{t}.Coeff + M{t}.Coeff;
                end
            end
            for t = 1:T
                models{t}.Coeff = models{t}.Coeff / k;
            end

            % save
            save(ammFilename, 'config', 'stim', 'resp', 'y', 'W', 'models');

            smm = SMM.load(ammFilename);
            smm.makePrediction();
            
            % Local functions
            function value = getField(i, field)
                smmFolder = SMM.getFoldFolder(i, procedureName, window, mask, k);
                smmFilename = fullfile(smmFolder, neuron);
                S = load(smmFilename, field);
                value = S.(field);
            end
        end
        
        function foldFolder = getFoldFolder(i, procedureName, window, mask, k)
            test = sprintf('%d %d', i, k);
            foldFolder = fullfile(...
                SMM.FOLDERS.ASSETS, ...
                SMM.getResultsFolder(procedureName, window, mask, test), ...
                'models');
        end
        
        function ammFolder = getAMMFolder(procedureName, window, mask, k)
            ammFolder = fullfile(SMM.FOLDERS.ASSETS, sprintf(...
                'procedure-%s-window-%d-mask-%d-test-%d/models', ...
                procedureName, window, mask, k));
        end
    end
    
    methods (Static)
        function plotAMMVsFolds(procedureName, window, mask, k)
            dllfoldsFilename = fullfile(...
                SMM.FOLDERS.ASSETS, ...
                SMM.getResultsFolder(procedureName, window, mask, num2str(k)), ...
                'dllfolds.mat');
            if ~exist(dllfoldsFilename, 'file')
                % folds
                dll.smm = [];
                fprintf('AMM vs. SMM\n');
                maintimer = tic();
                for i = 1:k % over folds
                    foldFolder = SMM.getFoldFolder(i, procedureName, window, mask, k);
                    fprintf('\n%d - %s:\n', i, foldFolder);
                    
                    dll.smm(i, :) = getDLLAllNeurons(foldFolder);
                end
                dll.smm = median(dll.smm);
                % avg
                ammFolder = SMM.getAMMFolder(procedureName, window, mask, k);
                dll.amm = getDLLAllNeurons(ammFolder);

                save(dllfoldsFilename, 'dll');
                toc(maintimer);
            end
            
            load(dllfoldsFilename, 'dll');
            
            SMM.createFigure('AMM vs. SMM');
            SMM.plotScatter(dll.smm, dll.amm, 'dll');
            xlabel('\DeltaLL/spk (bits/spk) [smm]');
            ylabel('\DeltaLL/spk (bits/spk) [amm]');
            folder = fileparts(dllfoldsFilename);
            saveas(gcf, fullfile(folder, 'dllfolds.png'));
            
            % Local functions
            function dll = getDLLAllNeurons(folder)
                listing = dir(fullfile(folder, '*.mat'));
                n = numel(listing);
                dll = zeros(1, n);
                parfor ind = 1:n
                    filename = fullfile(listing(ind).folder, listing(ind).name);
                    smm = SMM.load(filename);
                    dll(ind) = smm.getDLL('test', []);
                end
            end
        end
        
        function plotScatter(x, y, measure)
            % - Scatter

            scatter(x, y, 'filled');

            m = min(min(x), min(y));
            M = max(max(x), max(y));

            hold('on');
            line([m, M], [m, M], 'LineStyle', '--', 'Color', 'red', 'LineWidth', 2);

            % axis('equal');
            axis('square');

            % xticks(unique(round([m, median(x), M], 1)));
            % yticks(unique(round([m, median(y), M], 1)));

            % title('Negative Log-Likelihood');
            switch measure
                case 'dll'
                    xlabel('\DeltaLL/spk (bits/spk) [s-model]');
                    ylabel('\DeltaLL/spk (bits/spk) [smm]');
                case 'll'
                    xlabel('-LL (bits) [s-model]');
                    ylabel('-LL (bits) [smm]');
            end

            % set(gca, 'XScale', 'log');
            % set(gca, 'YScale', 'log');

            % SMM.setFontSize();
        end
    end
end

