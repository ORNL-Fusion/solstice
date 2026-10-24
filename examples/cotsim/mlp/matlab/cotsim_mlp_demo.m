% SOLSTICE COTSIM MLP demo (MATLAB), the counterpart of the SOLSTICE boundary_demo notebook.
% Run from the release folder:  >> cotsim_mlp_demo
% Loads the model, predicts the notebook's operating point, prints the COTSIM quantities, plots,
% writes cotsim_mlp_demo.json, compares with the Python reference (reference_demo.json) and times
% the model.
clear; h5 = 'solstice_cotsim_mlp.h5'; rho_c = 0.95;

tic; nn = read_NN_solstice(h5); t_load = toc;
params = struct('ptot', 1.299e7, 'puff_D2', 1.623e21, 'core_fueling', 1.0e20, 'dna', 0.3, 'chi', 0.3, ...
                'gamma_b_in', 9.312e20, 'gamma_b_out', 2.691e21, 'p_b_in', 68.47, 'p_b_out', 1.304e7);
y = nn.predict(params);
r = nn.result(y, rho_c, [], params);

fprintf('COTSIM boundary inputs:\n');
for nm = {'gamma_b_in', 'gamma_b_out', 'p_b_in', 'p_b_out'}
    fprintf('  %-12s = %.6e\n', nm{1}, params.(nm{1}));
end
fprintf('\nSOLSTICE MLP state at rho_c=%.3g:\n', rho_c);
units = struct('te', 'eV', 'ti', 'eV', 'ne', 'm^-3', 'ni', 'm^-3');
for nm = {'te', 'ti', 'ne', 'ni'}
    b = r.boundary_conditions.(nm{1});
    fprintf('  %-3s FSA=%.6e %s, OMP=%.6e %s\n', nm{1}, b.fsa, units.(nm{1}), b.omp, units.(nm{1}));
end
fprintf('\nP_rad = %.6e W\n', r.targets.P_rad_W);
for nm = {'inner_target', 'outer_target'}
    t = r.targets.(nm{1});
    fprintf('%s: q_peak = %.6e W/m^2, Te_strike = %.4g eV, ne_strike = %.4e m^-3\n', ...
            nm{1}, t.q_peak_W_m2, t.Te_strike_eV, t.ne_strike_m3);
end

plot_NN_solstice(r, 'cotsim_mlp_demo');
fid = fopen('cotsim_mlp_demo.json', 'w'); fprintf(fid, '%s', jsonencode(r, 'PrettyPrint', true)); fclose(fid);
fprintf('\nwrote cotsim_mlp_demo.json, cotsim_mlp_demo_profiles.png, cotsim_mlp_demo_targets.png\n');

% -- check against the Python reference of the same h5 file -------------------------------
if isfile('reference_demo.json')
    ref = jsondecode(fileread('reference_demo.json'));
    d = max(abs(y(:) - ref.outputs(:)) ./ abs(ref.outputs(:)));
    fprintf('max relative difference to the Python reference: %.2e (float32 weights: < 1e-5 expected)\n', d);
end

% -- timing ---------------------------------------------------------------------------------
n = 2000; x = nn.input_vector(params);
tic; for k = 1:n, nn.predict(x); end; t1 = toc / n;
X = repmat(x, 1000, 1); X(:, 9) = X(:, 9) .* linspace(0.5, 1.5, 1000)';
tic; Y = nn.predict(X); tb = toc / 1000;
tic; for k = 1:200, rr = nn.result(nn.predict(x), rho_c); end; tr = toc / 200;
fprintf('\nload %.3f s | predict, one case %.1f us | batch of 1000 %.2f us per case | predict + result %.1f us\n', ...
        t_load, 1e6 * t1, 1e6 * tb, 1e6 * tr);
