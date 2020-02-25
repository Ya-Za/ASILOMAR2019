function [] = amodel(session, channel, unit)
% Approximates the fitted S-kernels with Gaussian functions
%
% Parameters
% ----------
% - session: scalar
%   Session number with format: yymmdd
% - channel: scalar
%   Channel number

fprintf('===== A-Model =====\n\n');
main_timer = tic();

info = get_info();
num_folds = info.num_folds;

% `avg-models` directory
avg_models_folder = info.folders.avg_models;
if ~exist(avg_models_folder, 'dir')
    mkdir(avg_models_folder);
end

% filename
avg_model_filename = fullfile(avg_models_folder, get_filename(session, channel, unit));
if exist(avg_model_filename, 'file')
    return;
end

% first model
model = load_model(session, channel, 1);
profile = model;

% other models
for fold = 2:num_folds
    model = load_model(session, channel, unit, fold);

    profile.set_of_kernels.stm.knl = profile.set_of_kernels.stm.knl + model.set_of_kernels.stm.knl;
    profile.set_of_kernels.gas.knl = profile.set_of_kernels.gas.knl + model.set_of_kernels.gas.knl;
    profile.set_of_kernels.gas.cof = profile.set_of_kernels.gas.cof + model.set_of_kernels.gas.cof;
    profile.set_of_kernels.off.knl = profile.set_of_kernels.off.knl + model.set_of_kernels.off.knl;
    profile.set_of_kernels.psk.knl = profile.set_of_kernels.psk.knl + model.set_of_kernels.psk.knl;
end

% average
profile.set_of_kernels.stm.knl = profile.set_of_kernels.stm.knl / num_folds;
profile.set_of_kernels.gas.knl = profile.set_of_kernels.gas.knl / num_folds;
profile.set_of_kernels.gas.cof = profile.set_of_kernels.gas.cof / num_folds;
profile.set_of_kernels.off.knl = profile.set_of_kernels.off.knl / num_folds;
profile.set_of_kernels.psk.knl = profile.set_of_kernels.psk.knl / num_folds;

% save
save(avg_model_filename, '-struct', 'profile', '-v7.3')

fprintf('\n\n');
toc(main_timer);
end