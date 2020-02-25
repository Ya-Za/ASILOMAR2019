function save_figure(session, channel, unit, name)
% Save figure
%
% Parameters
% ----------
%
% - name: char vector
%   Name of saved figure

info = get_info();

folder = fullfile(...
    info.folders.results, ...
    get_id(session, channel, unit));
    
if ~isfolder(folder)
    mkdir(folder);
end

saveas(...
    gcf, ...
    fullfile(...
        folder,  ...
        [name, info.plotting.formattype]));
end