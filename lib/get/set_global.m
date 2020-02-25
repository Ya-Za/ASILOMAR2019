function set_global(name, value)
    global_filename = 'global.mat';
    S = struct(name, value);
    if isfile(global_filename)
        save(global_filename, '-struct', 'S', '-append');
    else
        save(global_filename, '-struct', 'S');
    end
end