function [profile] = load_profile(session, channel, unit)
% Load neural profile
%
% Parameters
% ----------
%
%
% Returns
% -------
% - profile: struct (todo: need more details)
%   Neural profile

info = get_info();

fullfilename = fullfile(...
    info.folders.profiles, ...
    get_filename(session, channel, unit));

profile = load_file(fullfilename);
end
