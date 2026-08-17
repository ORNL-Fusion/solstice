# =========================================================================================
# (C) (or copyright) 2026. UT-Battelle, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract DE-AC05-00OR22725 with
# UT-Battelle, LLC, which manages Oak Ridge National Laboratory (ORNL) for the U.S.
# Department of Energy (DOE). The U.S. Government is granted for itself and others acting
# on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
# to reproduce, prepare derivative works, distribute copies to the public, perform
# publicly and display publicly, and to permit others to do so. The DOE will provide
# public access to these results in accordance with the DOE Public Access Plan
# (http://energy.gov/downloads/doe-public-access-plan).
# =========================================================================================
# SPDX-License-Identifier: Apache-2.0
"""Deterministic post-processing of predicted fields into scalar QoIs and
consistency checks — never separate models (docs/specs/data_schema.md).

Planned: target 1D profiles (via face sets), peak target heat flux,
lambda_q (Eich fit), Psol, power-balance and particle-balance residuals
as standard evaluation metrics.
"""
