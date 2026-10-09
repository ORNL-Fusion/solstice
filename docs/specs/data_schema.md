# Canonical data schema (v0)

The contract between SOLPS output, training stores, and model loaders.
Everything in `solstice.data` serves this document.

## Principles

1. **Unstructured-first.** The atom is the *cell*, not `(iy, ix)`.
   GOAT wide-grid SOLPS produces genuinely unstructured polygonal meshes
   (`nCv` cells with variable face counts); structured grids are the
   special case where cells additionally carry `(ix, iy)` labels.
2. **SOLPS parsing follows SOLPS-routines.** Raw file reading
   (b2fgmtry, b2fstate, balance.nc), structured/unstructured detection
   (`nx,ny,ns` vs `nCi,nCg,nCv,nFc,nVx,nFs,nFt`), index conventions
   (jsep, cuts, guard cells) and region/face-set semantics are taken from
   the ORNL SOLPS-routines package — never re-derived here.
3. **Named variables with units. No positional channel conventions.**
4. **Graphs are derived, never stored.** Loaders build edges from face
   adjacency at load time.
5. Users of released weights never see this schema; it exists for
   training reproducibility and converters.

## Case layout (xarray-shaped; zarr for training stores, netcdf for exchange)

```
case/
  mesh/
    # cells (dim: cell, n_cells) — guard cells excluded, mapping retained
    cell_r, cell_z        [m]    cell centres
    cell_corners_r/z      [m]    (cell, max_corners) padded polygon corners
    cell_vol              [m^3]
    cell_region           [-]    region label (cvReg semantics from SOLPS-routines)
    cell_b                [T]    (cell, 4) B components (bb/cvBb convention)
    cell_psi_n            [-]    normalised psi (from vertex psi where available)
    cell_ix, cell_iy      [-]    OPTIONAL: structured indices (structured only)
  faces/
    # (dim: face, n_faces)
    face_vertices         (face, 2) vertex ids
    face_cells            (face, 2) adjacent cell ids (-1 = boundary)
    face_set              [-]   face-set label (targets, wall, core; fsFc semantics)
  vertices/
    vx_r, vx_z            [m]
    vx_psi                [Wb]  where available (vxFpsi)
  fields/      # per-cell named arrays (dim: cell), SI units
    te [eV], ti [eV], ne [m^-3], ua [m/s], ...
    # per-species variables use suffix naming: na_D0, na_D1, na_N3,
    # ua_D1, ... (species tag = element + charge state)
  derived/     # post-processed by SOLPS from the state.
    # Per-boundary-face arrays (dim: face, valid on target face sets):
    # deposited heat flux [W/m^2] and particle flux — the 1D target
    # load profiles. Kept in every dataset and learned as auxiliary
    # outputs (auxiliary heads), also the ground truth for target
    # evaluation metrics.
    # Per-cell arrays:
    # heat flux components (fhe*, fhi*), radiated power density
    # (rqrad_*, rqbrm_*), neutral pressure, ... Same naming rules.
    # Flagged derived because they are second-order: models may output
    # them directly (recommended, prior SPARC surrogate practice — enables
    # power-balance checks and denoises MC noise) or they can be
    # recomputed from predicted fields via solstice.physics.
  sources/     # per-cell named arrays (dim: cell)
    sp, sne, qe, qi, sm, dab2, dmb2, tab2, tmb2 (subset per case)
  inputs/      # scalar control parameters, named, with units in attrs.
    # Canonical names (converters map source metadata onto these; when
    # run metadata like params.json is regenerated, use these names
    # directly): pe, pi [W]; core_fueling (legacy params.json
    # key: core.density_m-3); puff_<gas> [atom/s]; dna, hci, hce [m^2/s]
  attrs:
    machine, topology ("structured" | "wide"), species list,
    solps_version, case_id, generation git hash, schema_version,
    convergence_status ("steady" | "oscillating" | "diverged" |
    "unclassified")
```

## Scalar quantities of interest

Scalar QoIs (peak target heat flux, lambda_q, Psol, power balance,
bolometer-like integrals) are never stored as primary data and never
get their own schema group: they are computed from fields/derived by
`solstice.physics` at evaluation time. Power balance between predicted
fields and the Psep input is a standard physics-consistency metric for
every released model.

## Topology dispatch in loaders

- **GNN**: edges = cell pairs sharing a face (+ face geometry as edge
  attrs). Identical for both topologies.
- **MLP**: per-cell rows. Both topologies.
- **U-Net**: assembles padded image from `cell_ix, cell_iy`. Structured
  cases only.

## Converters

- `from_solps`: SOLPS run directory -> canonical case, reading via
  SOLPS-routines (`solps_routines.readers`). Handles structured and
  wide/GOAT grids.
- `from_legacy_npz`: solpex-paper `coupling_dataset*.npz`
  (padded `(ny, nx)` images, fixed channel indices) -> canonical cases.
  Lossless: `(ix, iy)` retained as labels. Legacy channel-index maps live
  only inside this converter.

## Versioning

`schema_version` is stamped on every case. Breaking changes bump it and
get a migration note here.
