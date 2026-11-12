# Tomato experiment evidence

These directories preserve point-in-time experiment records. Code under `src/`,
`scripts/`, `tests/`, and `configs/` is the active implementation. Python and
configuration files inside archived reports describe the historical execution;
do not use them to update or launch new experiments.

- `tomato-20261111/`: E0 analysis, E1 startup/validation evidence and source hashes.
- `tomato-20261112-code-review/`: the pre-commit E2 draft review snapshot. Its
  statements about an uncommitted working tree describe that observation time.

The complete local archive also contains six `.tar.gz` source bundles and two
`.png` figures. They remain available in the project folder and experiment
storage but are excluded from Git. Manifests describe the complete local archive,
so a Git-only checkout intentionally does not contain every manifest entry.
The compact [baseline curve PDF](tomato-20261111/baseline-analysis/baseline_curves.pdf)
is included in Git.

Keep archived bytes unchanged: formatting, spelling and license-insertion hooks
exclude `reports/`, while JSON/YAML validity, file-size, case and private-key
checks still run. New source code belongs in the active package or scripts,
not in this archive. Existing `.log` files remain local under the repository's
log ignore rule; summaries and structured validation records are included.
