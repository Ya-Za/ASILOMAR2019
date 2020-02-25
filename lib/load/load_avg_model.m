function [model, fullfilename] = load_avg_model(session, channel, unit)
% Load average neural model
%
% Parameters
% ----------
%
%
% Returns
% -------
% - model: struct (todo: need more details)
%   Neural model
% - fullfilename: char vector
%   Filename of saved model

info = get_info();

fullfilename = fullfile(...
    info.folders.avg_models, ...
    get_filename(session, channel, unit));

model = load_file(fullfilename);
end
