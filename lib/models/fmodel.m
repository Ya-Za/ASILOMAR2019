function [] = fmodel(session, channel, unit)
% Approximates the fitted S-kernels with Gaussian functions
%
% Parameters
% ----------
% - session: scalar
%   Session number with format: yymmdd
% - channel: scalar
%   Channel number

fprintf('===== F-Model =====\n\n');
main_timer = tic();

info = get_info();
num_folds = info.num_folds;
lags = info.lags;

for fold = 1:num_folds
    [profile, filename] = load_model(session, channel, unit, fold);
    
    % lag intervals
    % todo: `set_of_params` -> `params`
    profile.set_of_params.lags = lags;
    
    % center of gaussians
    profile.centers = get_centers(session, channel, unit);
    
    % remove unnecessary data
    profile = rmfield(profile, {
        'set_of_data'
        'set_of_basis'
        'set_of_iterations'});
    
    % fit model
    fit_model(profile, filename);
end

fprintf('\n\n');
toc(main_timer);
end

function centers = get_centers(session, channel, unit)
% Get centers of Gaussians

data = load_main_lists();
probes = data.probe_list;

idx = (probes(:, 1) == session) & (probes(:, 2) == channel);

centers.RF = probes(idx, 3:4);
centers.FF = probes(idx, 5:6);
centers.ST = probes(idx, 7:8);
end

function [] = fit_model(profile, filename)
% Find optimum parameters

info = get_info();
foptions = info.foptions;
width = info.width;
height = info.height;
lags = info.lags;
times = info.times;

% center of gaussians
rf = profile.centers.RF;
ff = profile.centers.FF;
st = profile.centers.ST;

% stimulus kernel
stm_knl = profile.set_of_kernels.stm.knl;

% fixation kernel
idx_fix = -400 <= times & times <= -300;
knl_fix = squeeze(mean(stm_knl(:, :, idx_fix, :), 3));

% gaussian kernel = estimated version of stimulus kernel
gas_knl = zeros(size(stm_knl));
% gaussian parameters
gas_cof = nan(length(times), size(lags, 1), 25);

% time threshold
time_th = 100;
for it = 1:length(times)
    t = times(it);
    fprintf('%03g ms from saccade.\n', t);
    
    knl = squeeze(stm_knl(:, :, it, :));
    new_knl = nan(size(knl));
    
    for il = 1:size(lags, 1)
        ind_lag = lags(il, 1):lags(il, 2);
        
        fun = @(x) loss_fun(...
            knl_fix(:, :, ind_lag), ...
            knl(:, :, ind_lag), ...
            x, ...
            width, ...
            height);

        if t < time_th
            x0 = [1 rf .5 .5 0 0 0 1 ff .5 .5 0 0 0 1 st .5 .5 0 0 0 0];

            lb = [-inf rf-1 0 0 -1 -5 -5 -inf ff-1 0 0 -1 -5 -5 -inf st-1 0 0 -1 -5 -5 -inf];
            ub = [+inf rf+1 2 2 +1 +5 +5 +inf ff+1 2 2 +1 +5 +5 +inf st+1 2 2 +1 +5 +5 +inf];
        else
            x0 = [1 rf .2 .2 0 0 0 1 ff .5 .5 0 0 0 1 st .5 .5 0 0 0 0];

            lb = [-inf rf-0 0 0 -1 -5 -5 -inf ff-1 0 0 -1 -5 -5 -inf st-1 0 0 -1 -5 -5 -inf];
            ub = [+inf rf+0 1 1 +1 +5 +5 +inf ff+1 2 2 +1 +5 +5 +inf st+1 2 2 +1 +5 +5 +inf];
        end
        
        % todo: use gradient of objective function to get more fast/robust
        % solution
        best_params = fmincon(fun, x0, [], [], [], [], lb, ub, [], foptions);
        
        new_knl(:, :, ind_lag) = obj_fun(...
            knl_fix(:, :, ind_lag), ...
            best_params(1:8), ...
            best_params(9:16), ...
            best_params(17:24), ...
            best_params(25), ...
            width, ...
            height);
        
        gas_cof(it, il, :) = best_params;
    end
    
    for x = 1:width
        for y = 1:height
            % todo: `10` -> `window`
            new_knl(x, y, :) = smooth(squeeze(new_knl(x, y, :)), 10);
        end
    end
    
    gas_knl(:, :, it, :) = new_knl;
end

profile.set_of_kernels.gas.knl = gas_knl;
profile.set_of_kernels.gas.cof = gas_cof;

% save f-kernels
save(filename, '-struct', 'profile', '-append')
end

function [loss] = loss_fun(knl_fix, knl, params, width, height)
% Loss function

new_knl = obj_fun(...
    knl_fix, ...
    params(1:8), ...
    params(9:16), ...
    params(17:24), ...
    params(25), ...
    width, ...
    height);

% todo: 
%   - `fmincon` -> `lsqnonlin`
%   - `norm` -> `Frobenius norm`
loss = norm(new_knl(:) - knl(:));
end

function [y] = obj_fun(x0, rf, ff, st, c, width, height)
% Objective function

sz = [1, 1, size(x0, 3)];
y = x0 + c + ...
    repmat(skew_gaussian(rf(1), rf(2), rf(3), rf(4), rf(5), rf(6), rf(7), rf(8), width, height), sz) + ...
    repmat(skew_gaussian(ff(1), ff(2), ff(3), ff(4), ff(5), ff(6), ff(7), ff(8), width, height), sz) + ...
    repmat(skew_gaussian(st(1), st(2), st(3), st(4), st(5), st(6), st(7), st(8), width, height), sz);
end

function [z] = skew_gaussian(a, mx, my, sx, sy, rho, alpha, beta, width, height)
% Skewed Gaussian kernels
% todo: instead of skew gaussian we can just use 2d gaussian

[y, x] = meshgrid(1:height, 1:width);

% todo: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
% todo: phi = mvnpdf([mx, my], [sx*sx, rho*sx*sy; rho*sx*sy, sy*sy]);
phi = ...
    a .* exp(-1 ./ 2 ./ (1 - rho .^ 2) .* (...
        (x - mx) .^ 2 ./ sx .^ 2 + ...
        (y - my) .^ 2 ./ sy .^ 2 - ...
        2 .* rho .* (x - mx) .* (y - my) ./ sx ./ sy));

% todo: https://en.wikipedia.org/wiki/Skew_normal_distribution
PHI = normcdf(alpha .* (x - mx)) .* normcdf(beta .* (y - my));

z(:, :, 1) = phi .* PHI;
end
