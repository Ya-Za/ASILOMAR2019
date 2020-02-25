function [model, fullfilename] = load_model(session, channel, unit, fold)
% Load neural model for a given fold number
%
% Parameters
% ----------
%
% - fold: integer scalar
%   Fold number
%
% Returns
% -------
% - model: struct (todo: need more details)
%   Neural model
% - fullfilename: char vector
%   Filename of saved model

info = get_info();

fullfilename = fullfile(...
    info.folders.models, ...
    get_id(session, channel, unit), ...
    sprintf('fold%02d.mat', fold));

model = load_file(fullfilename);
end
