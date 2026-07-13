# Contributing

We welcome focused fixes and reproducibility improvements for the released
RealEstate10K pipeline.

1. Open an issue before starting a large change.
2. Keep training and sampling behavior compatible with the released
   checkpoints. Call out any intentional numerical change explicitly.
3. Run `python -m compileall -q .` and verify that all targets in
   `configs/live_re10k.yaml` import successfully.
4. Do not commit datasets, checkpoints, credentials, generated videos, or
   experiment logs.
5. Submit a pull request with the motivation, validation command, hardware,
   and observed behavior.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0.
