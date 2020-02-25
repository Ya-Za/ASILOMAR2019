function h = create_figure(name)
    % Create `full screen` figure
    %
    % Parameters
    % ----------
    % - name: string
    %   Name of figure
    %
    % Return
    % - h: matlab.ui.Figure
    %   Handle of created figure
    
    if nargin < 1
        name = '';
    end
    
    h = figure(...
        'Name', name, ...
        'Color', 'white', ...
        'NumberTitle', 'off', ...
        'Units', 'normalized', ...
        'OuterPosition', [0, 0, 1, 1] ...
    );
end

