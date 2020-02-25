function [data, fullfilename] = load_main_lists()
% Load neural model for a given fold number
%
% Returns
% -------
% - data: struct (todo: need more details)
%   Data of main lists
% - fullfilename: char vector
%   Filename of saved model

info = get_info();

fullfilename = fullfile(...
    info.folders.data, ...
    'main_lists');

data = load_file(fullfilename);
end
