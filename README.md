# Focal neuronal loss and cell replacement in a connectome model of fly steering

Code and data for:

Habib P, Petr M, Rust R. *Focal neuronal loss and cell replacement in a connectome model of fly steering.* Manuscript in preparation (2026).

The study simulates 165,122 neurons of the male *Drosophila* central nervous system (MaleCNS v1.0 connectome) with a leaky integrate-and-fire model, removes neurons at defined locations, adds replacement cells under different selection, identity and wiring rules, and converts wing motor-neuron output into steering in a reduced two-dimensional movement controller.

This repository contains only the final code and data used for the manuscript figures, supplementary figures and movies. Earlier drafts, exploratory pilots and superseded analyses are not included.

## Contents

| Folder | Contents |
|---|---|
| `code/v5` | Neural network simulator (`flysim.py`, `flysim_fast.py`), stimuli and readouts (`experiments.py`), replacement wiring (`graft_typed.py`), movement controller (`flight.py`, `flight_fast.py`), decoder calibration (`calibrate.py`), connectome import (`restore_connectivity.py`) and the 56-site screen runner (`run_states.py`) |
| `code/v8` | Main simulation batch, extensions, analysis, figures and Movie 2 |
| `code/v8/journal_movies` | Renderer for Supplementary Movies 1, 3 and 4 |
| `code/v7` | Path smoothing and 3D helpers imported by the movie code |
| `code/literature_benchmarks_20261021` | Model counterparts of published perturbations (Supplementary Fig. 2) |
| `figures/v8/review_20261012` | Builder for the Figure 2a activity panel |
| `data/v5` | Model inputs derived from MaleCNS v1.0: signed connectivity matrix, neuron table, cell groups, mirror map and steering decoder |
| `results/v5` | Decoder calibration and the 56-site, three-size lesion screen (Fig. 1b,c) |
| `results/v8` | Final simulation batch: raw states, per-realization readouts, summaries, paired contrasts, example paths, lesion masks, replacement positions and protocols |
| `results/literature_benchmarks_20261021` | Simulations behind Supplementary Fig. 2 |
| `source_data` | Figure 2 and Supplementary Fig. 2 source tables, Movie 2 spike recordings and Movie 1, 3, 4 payloads |
| `figures` | Final main and supplementary figures (PDF, PNG) |
| `movies` | Final Supplementary Movies 1 to 4 and captions |
| `tests` | Simulator tests |

## Main result files

- `results/v8/integrated_20261012/per_realization.csv`: one row per condition and realization (79 conditions x 100 realizations). Movement endpoints are averaged over 20 matched wind-gust trials per realization.
- `results/v8/integrated_20261012/summary.csv`: mean, SD, SEM, 95% confidence limits and 2.5th/97.5th percentiles per condition and readout.
- `results/v8/integrated_20261012/gust_trials.csv`: all 158,000 movement trials.
- `results/v8/tables/paired_contrasts.csv`: differences between matched conditions within realization.
- `results/v8/tables/graft_composition_and_connections.csv`: cell counts, identity matches, donor use and connection counts for every graft.
- `results/v8/trajectories/`: the example paths shown in the figures and movies (neural seed 1000, intervention seed 2000, gust seed 3).

Realizations are stochastic replicates of one connectome with fixed parameters (neural seeds 1000-1099, intervention seeds 2000-2099). They describe variability within the model, not biological replication.

### Condition names

| Name | Meaning |
|---|---|
| `intact` | No lesion |
| `lesion20_N`, `lesion25_N` | N neurons removed at site 20 or site 25 |
| `selective_*` | Removal of defined cell classes within the 1,000-cell site-20 mask |
| `selection_random_N`, `selection_interface_N`, `selection_ranked_N` | N lost cells restored with original identities and partners, chosen at random, by projection priority or by motor-control priority (2,000-cell lesion) |
| `complete_original_2000` | All 2,000 lost cells restored with original partners |
| `connections_random_partners_100`, `review_random_partners_dose_N`, `review_full_random_partners_2000` | Intended identities with random nearby partners |
| `connections_correct_types_100`, `typed_dose_correct_types_N` | Matched cell types, partners copied from a surviving cell of that type |
| `connections_incorrect_types_100`, `typed_dose_incorrect_types_N` | Shuffled lost types (Supplementary Fig. 4c) |
| `non_target_types_N` | Central-brain types absent from the lesion, partners copied by type |

## How the results were produced

Scripts locate files relative to the repository root, which mirrors the original project layout.

1. **Model inputs.** `code/v5/restore_connectivity.py` builds `data/v5` from the MaleCNS v1.0 release. The full edge table (`data/v5/traced_connections.parquet`, 132 MB) exceeds the GitHub file limit and is not included; it is regenerated from the Janelia FlyEM MaleCNS v1.0 download. The simulator reads `W_csc.npz`, which is included.
2. **Decoder calibration.** `code/v5/calibrate.py` writes `data/v5/decoder.npz` and `results/v5/reference.json`.
3. **56-site lesion screen.** `code/v5/run_states.py results/v5/specs_states.json results/v5/states.jsonl`; site coordinates are in `results/v5/lesion_sites.csv` and summaries in `results/v5/revision_20261002`.
4. **Main batch.** `code/v8/run_confirmatory.py` (protocol `results/v8/prespecified_protocol.json`) writes `results/v8/states.jsonl`. Extensions: `run_typed_dose.py` (matched and shuffled types across doses), `run_non_target.py` (non-target types) and `integrate_approved_fig2.py` (random-wiring doses and integration into `results/v8/integrated_20261012`). `analyze.py` derives readouts and summaries. `revise_supplement.py` produces the gust and supplementary control tables in `results/v8/supplement_revision_20261012`.
5. **Published perturbations.** `code/literature_benchmarks_20261021/run_benchmarks.py` and `build_report.py`.
6. **Figures.** `code/v8/build_figures.py`; Figure 2a activity panel `figures/v8/review_20261012/build_fig2a_activity_review.py` with recordings from `code/v8/record_fig2a_profiles.py`.
7. **Movies.** Movie 2: `code/v8/curated_activity_movie.py` then `trim_activity_movie.py`. Movies 1, 3 and 4: see `code/v8/journal_movies/README.txt`. Movies replay saved paths and recordings; they do not rerun simulations.

## Requirements

Python 3.11 or 3.12 with numpy, scipy, pandas, pyarrow, numba and matplotlib (see `requirements.txt`). The full batch was run on multi-core CPUs. Movie rendering additionally needs Node.js with Playwright (Chromium) and ffmpeg.

## Licenses

Code: MIT (`LICENSE`). Derived data, figures and movies: CC BY 4.0 (`LICENSE-DATA`). The connectome itself is from the MaleCNS v1.0 release by Janelia Research Campus FlyEM and collaborators (Berg et al., *Cell* 2026); please cite it when using the model inputs.

## Contact

Ruslan Rust, Zilkha Neurogenetic Institute, Keck School of Medicine, University of Southern California (rrust@usc.edu)
