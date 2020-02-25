function [map] = load_map(session, channel, unit, prb)
% Load map of bases for a given probe
%
% Parameters
% ----------
%
% - prb: scalar
%   Probe index
%
% Returns
% -------
% - map: array(time, delay, iteration)
%   Map of bases in time/delay grid

info = get_info();

fullfilename = fullfile(...
    info.folders.bases, ...
    get_id(session, channel, unit), ...
    sprintf('prb%02d.mat', prb));

file = load_file(fullfilename);
map = file.map;
end
