# Minjoo Brain-pong task

`brain_pong_inputs/` is a complete copy of the local
`brain-pong-inputs/brain_pong_inputs/` package. The two package trees are kept
structurally and byte-for-byte identical so updates can be copied in either
direction.

The DecNef integration is intentionally split as follows:

- `rt_pipeline.py`: realtime fMRI preprocessing and decoder scoring
- `brainpong_adapter.py`: reference z-score to signed feedback and paddle dynamics
- `task.Minjoo.brain_pong_inputs`: shared Pong, TrialBlock, leveling, config, and logs
- `rt_brainpong_parallel.py`: scanner trigger, baseline, trials, ITIs, queue input,
  and run-level logging

The same package supports `mouse`, `press`, `wheel`, and `brain` input. The
standalone CLI uses its original loop. The scanner runner injects a neural
controller through the shared optional callback interfaces while preserving the
same TrialBlock leveling and YAML configuration.

Current scanner schedule after trigger:

- 30-second initial baseline
- six 120-second feedback trials
- five 6-second ITIs
- total: 780 seconds (13 minutes)

Both standalone and scanner runs write `blockXX.tsv`, `blockXX_trialXX.tsv`,
and `blockXX.log` using the same shared implementation.
