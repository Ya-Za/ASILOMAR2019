function plot_time_delay_map(map)
% Plot time delay sensitivity map
%
% Parameters
% ----------
% - map: matrix
%   (time x delay) sensitivity map

info = get_info();
times = info.times;
[T, D] = size(map);

% surf(map');
% view([0, 90]);
% shading('interp');

imagesc(map');
axis('xy');

colorbar('XTick', unique(round([min(map(:)), max(map(:))], 2)));

xlabel('Time form saccade onset (ms)');
ylabel('Delay (ms)');

tidx = [1, ceil(T / 2), T];
xticks(tidx);
xticklabels(string(times(tidx)));

yticks([1, 50:50:D]);

set_axis();
end