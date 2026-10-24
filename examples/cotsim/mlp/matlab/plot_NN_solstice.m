function figs = plot_NN_solstice(r, outprefix)
% Plots of a read_NN_solstice result (r = nn.result(y, rho_c)), as in the SOLSTICE
% boundary_demo notebook: OMP and flux-surface-averaged Te, Ti, ne, ni versus psi_N with rho_c,
% and the heat-flux profiles on the inner and outer targets. (The notebook's 2D maps need the
% full 2D state, which the MLP does not predict.)
%
%   figs = plot_NN_solstice(r);                 % two figures
%   plot_NN_solstice(r, 'cotsim_mlp');          % also saves cotsim_mlp_profiles.png, cotsim_mlp_targets.png
if nargin < 2, outprefix = ''; end
p = r.profiles; x = p.x; closed = logical(p.closed);
labels = {'te', 'T_e [eV]'; 'ti', 'T_i [eV]'; 'ne', 'n_e [m^{-3}]'; 'ni', 'n_i [m^{-3}]'};

figs(1) = figure('Name', 'SOLSTICE MLP profiles', 'Position', [50 50 1600 420]);
for k = 1:4
    ax = subplot(1, 4, k); hold(ax, 'on'); box(ax, 'on');
    plot(ax, x, p.([labels{k, 1} '_omp']), '.-', 'Color', [0.55 0.55 0.55], 'DisplayName', 'OMP');
    fsa = p.([labels{k, 1} '_fsa']);
    plot(ax, x(closed), fsa(closed), 'o-', 'MarkerSize', 4, 'DisplayName', 'FSA');
    xline(ax, r.rho_c, '--r', 'DisplayName', '\rho_c');
    xline(ax, 1.0, '-k', 'HandleVisibility', 'off');
    xlabel(ax, '\psi_N'); ylabel(ax, labels{k, 2}); title(ax, strtok(labels{k, 2}));
    if startsWith(labels{k, 1}, 'n'), set(ax, 'YScale', 'log'); end
    if k == 1, legend(ax, 'Location', 'best'); end
end
sgtitle(sprintf('SOLSTICE MLP state supplied to COTSIM at \\rho_c = %.3g', r.rho_c));

figs(2) = figure('Name', 'SOLSTICE MLP targets', 'Position', [50 520 1100 420]);
sides = {'inner_target', 'Inner target'; 'outer_target', 'Outer target'};
for k = 1:2
    ax = subplot(1, 2, k); hold(ax, 'on'); box(ax, 'on');
    t = r.targets.(sides{k, 1});
    semilogy(ax, t.iy, t.q_profile_W_m2, 'o-', 'MarkerSize', 4);
    plot(ax, t.iy_peak, t.q_peak_W_m2, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    xline(ax, r.profiles.iy(find(~closed, 1)) - 0.5, '-k');
    set(ax, 'YScale', 'log'); xlabel(ax, 'iy (radial cell index, separatrix = line)');
    ylabel(ax, '|q_{pol}| [W m^{-2}]');
    title(ax, sprintf('%s: q_{peak} %.3g W/m^2, T_{e,strike} %.3g eV, n_{e,strike} %.3g m^{-3}', ...
          sides{k, 2}, t.q_peak_W_m2, t.Te_strike_eV, t.ne_strike_m3), 'FontWeight', 'normal');
end
sgtitle(sprintf('Target heat flux, P_{rad} = %.3g W', r.targets.P_rad_W));

if ~isempty(outprefix)
    saveas(figs(1), [outprefix '_profiles.png']);
    saveas(figs(2), [outprefix '_targets.png']);
end
end
