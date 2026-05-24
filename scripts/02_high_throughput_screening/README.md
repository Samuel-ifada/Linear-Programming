# High-Throughput Screening

Scripts for generating or enumerating high-throughput alloy screening results.

## Files

- `generate_lp_grid.py`: Builds the FLARE linear-programming constraint system, enumerates the 5 at.% feasible grid, and writes survivor CSV files and polytope figures.

## Current data assumptions

`generate_lp_grid.py` expects `EUROFER97_flare_DS.csv` to be available alongside the script unless the script paths are adjusted.
