function [B] = make_bsplie_bases(L, knots)
    % Draw, manipulate and reconstruct B-splines.
    % version 1.5.0.0 (21.1 KB) by Levente Hunyadi
    % link: https://www.mathworks.com/matlabcentral/fileexchange/27374-b-splines

    n = 3; % order = 2
    t = min(knots):max(knots); B = [];
    for index = 0:numel(knots)-n-1
        [y, ~] = bspline_basis(index, n, knots, t);
        B = [ B ; y ];
    end
    I_selected = min(L) <= t & t <= max(L);
    B = B(:, I_selected==1);
    end

    function [y, x] = bspline_basis(j, n, t, x)
    validateattributes(j, {'numeric'}, {'nonnegative', 'integer', 'scalar'});
    validateattributes(n, {'numeric'}, {'positive', 'integer', 'scalar'});
    validateattributes(t, {'numeric'}, {'real', 'vector'});
    assert(all( t(2:end)-t(1:end-1) >= 0 ), ...
        'Knot vector values should be nondecreasing.');
    if nargin < 4
        x = linspace(t(n), t(end-n+1), 100);
    else
        validateattributes(x, {'numeric'}, {'real', 'vector'});
    end
    assert(0 <= j && j < numel(t)-n, ...
        'Invalid interval index j = %d, expected 0 =< j < %d (0 =< j < numel(t)-n).', j, numel(t)-n);
    y = bspline_basis_recurrence(j, n, t, x);
    end

    function [y] = bspline_basis_recurrence(j, n, t, x)
    y = zeros(size(x));
    if n > 1
        b = bspline_basis(j, n-1, t, x);
        dn = x - t(j+1);
        dd = t(j+n) - t(j+1);
        if dd ~= 0  % indeterminate forms 0/0 are deemed to be zero
            y = y + b.*(dn./dd);
        end
        b = bspline_basis(j+1, n-1, t, x);
        dn = t(j+n+1) - x;
        dd = t(j+n+1) - t(j+1+1);
        if dd ~= 0
            y = y + b.*(dn./dd);
        end
    elseif t(j+2) < t(end)
        y(t(j+1) <= x & x < t(j+2)) = 1;
    else
        y(t(j+1) <= x) = 1;
    end
end
