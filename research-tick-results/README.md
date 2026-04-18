# research-tick-results

Single root for every artifact emitted by `/research-tick`. Layout:

- `ticks/<ts>/` — one directory per tick invocation. Contains `report.md`,
  `proposals.json`, and `digests/<project>/{digest.json,digest.csv,digest_plots/}`.
- `probes/<ts>/` — short probe runs submitted before a full verification stage,
  kept separate so they don't pollute the stage digest.
- `verification/<Hxx>/<stage>/` — user-facing gate artifacts: `key_figure.png`
  and `DECISION.md`. Review these to authorize advancing a hypothesis.
- `drafts/` — staging area for auto-generated LaTeX.
- `archive/` — legacy `reports/`, `plots/`, `digests/` from before
  consolidation, preserved for reference.
- `budget.yaml` — autonomy fence consumed by `research_tick.py`. Edit this to
  change which hyperparameters the tick may auto-submit.

Contents of every subdirectory are gitignored; only `budget.yaml` and
`README.md` are versioned.
