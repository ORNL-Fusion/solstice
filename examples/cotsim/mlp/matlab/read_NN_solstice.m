classdef read_NN_solstice
% SOLSTICE COTSIM MLP reader (MATLAB). Adapted from the COTSIM read_NN class: same h5 walk
% (model_weights/<layer>/<layer>/vars/{kernel,bias}) and dense + ReLU forward pass. The
% normalization, the log10 flags, names, units and mesh constants are read from the same file,
% and predict() applies the input/output transforms itself.
%
%   nn  = read_NN_solstice('solstice_cotsim_mlp.h5');
%   p   = struct('chi',0.3,'core_fueling',1e20,'dna',0.3,'gamma_b_in',9.312e20, ...
%                'gamma_b_out',2.691e21,'p_b_in',68.47,'p_b_out',1.304e7,'ptot',1.299e7,'puff_D2',1.623e21);
%   y   = nn.predict(p);            % 1 x 295 physical (also accepts a 1x9 vector in input_names order,
%                                   %   or an N x 9 matrix -> N x 295)
%   r   = nn.result(y, 0.95);       % struct like SOLSTICE cotsim_boundary_demo.json:
%                                   %   boundary_conditions.te.fsa/.omp ..., targets, profiles
%   plot_NN_solstice(r);            % profile and target plots
%
% Inputs: chi, dna [m^2/s]; core_fueling, puff_D2, gamma_b_in, gamma_b_out [1/s]; p_b_in, p_b_out, ptot [W].
% rho_c is normalised poloidal flux psi_N. Without psi_sep - psi_axis the innermost mesh row is
% pinned to rho_c (as in SOLSTICE examples/cotsim/boundary.py); pass it to result()/at_rho() otherwise.

    properties
        weights
        biases
        n_layers
        x_mean
        x_std
        x_log
        y_mean
        y_std
        y_log
        input_names
        output_names
        output_units
        mesh
        idx            % output indices per block
    end

    methods
        function obj = read_NN_solstice(h5file)
            info = h5info(h5file, '/model_weights');
            layers = sort(arrayfun(@(g) g.Name, info.Groups, 'UniformOutput', false));
            obj.n_layers = numel(layers);
            obj.weights = cell(obj.n_layers, 1); obj.biases = cell(obj.n_layers, 1);
            for i = 1:obj.n_layers
                [~, nm] = fileparts(layers{i});
                base = [layers{i} '/' nm '/vars'];
                obj.weights{i} = double(h5read(h5file, [base '/kernel']));   % h5read transposes: (n_out, n_in)
                obj.biases{i}  = double(h5read(h5file, [base '/bias']));     % (n_out, 1)
            end
            rd = @(p) double(h5read(h5file, p));
            obj.x_mean = rd('/normalization/x_mean'); obj.x_std = rd('/normalization/x_std');
            obj.x_log  = rd('/normalization/x_log') == 1;
            obj.y_mean = rd('/normalization/y_mean'); obj.y_std = rd('/normalization/y_std');
            obj.y_log  = rd('/normalization/y_log') == 1;
            obj.input_names  = read_names(h5file, '/normalization/input_names');
            obj.output_names = read_names(h5file, '/normalization/output_names');
            obj.output_units = read_names(h5file, '/normalization/output_units');
            for nm = {'rows', 'closed_rows', 'dpsi', 'R_omp', 'iy_sep', 'iy_core_last', 'R_sep_omp'}
                obj.mesh.(nm{1}) = reshape(rd(['/solstice_mesh/' nm{1}]), 1, []);
            end
            obj.mesh.closed = ismember(obj.mesh.rows, obj.mesh.closed_rows);
            for f = {'te', 'ti', 'ne', 'ni'}
                obj.idx.([f{1} '_fsa']) = find(strncmp(obj.output_names, [f{1} '_fsa_'], numel(f{1}) + 5));
                obj.idx.([f{1} '_omp']) = find(strncmp(obj.output_names, [f{1} '_omp_'], numel(f{1}) + 5));
            end
            obj.idx.q_outer = find(strncmp(obj.output_names, 'q_outer_', 8));
            obj.idx.q_inner = find(strncmp(obj.output_names, 'q_inner_', 8));
            for nm = {'q_peak_outer', 'q_peak_inner', 'Te_strike_outer', 'ne_strike_outer', ...
                      'Te_strike_inner', 'ne_strike_inner', 'P_rad'}
                obj.idx.(nm{1}) = find(strcmp(obj.output_names, nm{1}));
            end
            obj.idx.iy_outer = cellfun(@(s) sscanf(s, 'q_outer_%d'), obj.output_names(obj.idx.q_outer))';
            obj.idx.iy_inner = cellfun(@(s) sscanf(s, 'q_inner_%d'), obj.output_names(obj.idx.q_inner))';
        end

        function x = input_vector(obj, p)
            % struct with fields named as input_names -> 1 x n_in; numeric input passes through.
            if isstruct(p)
                x = zeros(1, numel(obj.input_names));
                for j = 1:numel(obj.input_names)
                    x(j) = p.(obj.input_names{j});
                end
            else
                x = double(p);
                if isvector(x), x = reshape(x, 1, []); end
            end
        end

        function outputs = predict(obj, inputs)
            % inputs: struct, 1 x n_in vector, or N x n_in matrix (input_names order, physical units)
            % -> N x n_out physical outputs (output_names / output_units order).
            X = obj.input_vector(inputs);
            if size(X, 2) ~= numel(obj.x_mean)
                error('read_NN_solstice:predict', 'expected %d inputs per case, got %d', numel(obj.x_mean), size(X, 2));
            end
            val = X';                                       % n_in x N
            val(obj.x_log, :) = log10(val(obj.x_log, :));
            val = (val - obj.x_mean) ./ obj.x_std;
            for i = 1:obj.n_layers
                val = obj.weights{i} * val + obj.biases{i};
                if i < obj.n_layers
                    val = max(val, 0);                      % ReLU
                end
            end
            val = val .* obj.y_std + obj.y_mean;
            val(obj.y_log, :) = 10 .^ val(obj.y_log, :);
            outputs = val';
        end

        function s = unpack(obj, y)
            % One output row -> named blocks (profiles core -> wall, plates sorted by iy).
            s = struct();
            for f = fieldnames(obj.idx)'
                if ~startsWith(f{1}, 'iy_')
                    s.(f{1}) = y(obj.idx.(f{1}));
                end
            end
        end

        function psin = psi_n(obj, rho_c, psi_axis_to_sep)
            % psi_N per outer-midplane row (all 36 rows, core -> wall).
            if nargin < 3 || isempty(psi_axis_to_sep)
                span = obj.mesh.dpsi(1) / (rho_c - 1.0);   % pin the innermost row to rho_c
            else
                span = abs(psi_axis_to_sep);
            end
            psin = 1.0 + obj.mesh.dpsi / span;
        end

        function bc = at_rho(obj, y, rho_c, psi_axis_to_sep)
            % Te, Ti [eV], ne, ni [m^-3] at rho_c: flux-surface average (.fsa) and outer-midplane (.omp).
            if nargin < 4, psi_axis_to_sep = []; end
            x = obj.psi_n(rho_c, psi_axis_to_sep);
            xc = x(obj.mesh.closed);
            if rho_c < min(xc) - 1e-9 || rho_c > max(xc) + 1e-9
                error('read_NN_solstice:at_rho', 'rho_c=%g outside the closed rows (%g .. %g)', rho_c, min(xc), max(xc));
            end
            s = obj.unpack(y); bc = struct();
            for f = {'te', 'ti', 'ne', 'ni'}
                omp = s.([f{1} '_omp']);
                bc.(f{1}) = struct('fsa', interp1(xc, s.([f{1} '_fsa']), rho_c, 'linear'), ...
                                   'omp', interp1(xc, omp(obj.mesh.closed), rho_c, 'linear'));
            end
        end

        function r = result(obj, y, rho_c, psi_axis_to_sep, params)
            % Same content as SOLSTICE examples/cotsim/cotsim_boundary_demo.json (no 2D fields).
            if nargin < 4, psi_axis_to_sep = []; end
            if nargin < 5, params = []; end
            s = obj.unpack(y); x = obj.psi_n(rho_c, psi_axis_to_sep);
            r = struct();
            r.model = 'SOLSTICE COTSIM MLP (pepc-diiid-cotsim-677-mlp)';
            r.rho_c = rho_c;
            if ~isempty(params), r.cotsim_boundary_inputs = params; end
            r.boundary_conditions = obj.at_rho(y, rho_c, psi_axis_to_sep);
            r.targets.P_rad_W = s.P_rad;
            sides = {'inner_target', 'inner'; 'outer_target', 'outer'};
            for k = 1:2
                q = s.(['q_' sides{k, 2}]); iy = obj.idx.(['iy_' sides{k, 2}]);
                [~, kp] = max(q);
                r.targets.(sides{k, 1}) = struct( ...
                    'q_peak_W_m2', s.(['q_peak_' sides{k, 2}]), 'iy_peak', iy(kp), ...
                    'Te_strike_eV', s.(['Te_strike_' sides{k, 2}]), 'ne_strike_m3', s.(['ne_strike_' sides{k, 2}]), ...
                    'q_profile_W_m2', q, 'iy', iy);
            end
            p = struct('iy', obj.mesh.rows, 'x', x, 'R_omp', obj.mesh.R_omp, 'dpsi', obj.mesh.dpsi, ...
                       'closed', obj.mesh.closed);
            for f = {'te', 'ti', 'ne', 'ni'}
                fsa = nan(1, numel(obj.mesh.rows)); fsa(obj.mesh.closed) = s.([f{1} '_fsa']);
                p.([f{1} '_fsa']) = fsa; p.([f{1} '_omp']) = s.([f{1} '_omp']);
            end
            r.profiles = p;
            r.ni_definition = 'ni = na_D1';
        end
    end
end

function names = read_names(h5file, path)
    v = h5read(h5file, path);
    if ischar(v), v = cellstr(v'); end
    names = cellfun(@(s) deblank(char(s)), cellstr(v), 'UniformOutput', false);
    names = reshape(names, [], 1);
end
