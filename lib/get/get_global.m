function value = get_global(name)
    global_filename = 'global.mat';
    if isfile(global_filename)
        S = load('global.mat', name);
        value = S.(name);
    else
        value = [];
    end
end