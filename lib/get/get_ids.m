function ids = get_ids()
    % Get neuron's IDs
    %
    % Returns
    % -------
    % - id: cell array of char vectors
    %   Neuron's IDs saved on the `data` folder

    info = get_info();
    
    listing = dir(fullfile(info.folders.data, '*.mat'));
    
    n = numel(listing);
    ids = cell(n, 1);
    for i = 1:n
        [~, ids{i}] = fileparts(listing(i).name);
    end
end