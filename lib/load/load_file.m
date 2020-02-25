function file = load_file(filename)
    % Load file
    %
    % Parameters
    % ----------
    % - filename: char vaector
    %   Filename which would be loaded
    %
    % Returns
    % -------
    % - data: structure array
    %   Loaded data


    fprintf('Load `%s`: ', filename);

    tic();
    file = load(filename);
    toc();
end
