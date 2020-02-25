function save_error(ME)
    warning(ME.message);
    file = fopen('error.txt', 'a');
    fprintf(file, '%s\n%s\n\n', ME.identifier, ME.message);
    fclose(file);
end