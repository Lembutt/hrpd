# HRPD — example repository

A minimal example of **HRPD (Healthy-Referenced Physics-informed Diagnosis)** - synthetic-to-real bearing fault diagnosis trained without a single real fault sample.

- `hrpd/` - the documented reference implementation (features, models, training/adaptation/inference pipeline, metrics).
- `example.ipynb` - an end-to-end walkthrough on a bundled miniature MFPT dataset: inspect the envelope-order spectra, build the two-channel residual input and the wide-window physics feature, train HRPD with a short schedule, and evaluate on real faults.
- `data/mfpt_base5_data/` - the MFPT benchmark at one synthetic-source seed (the real target-healthy and test splits are complete).

## Quick start

```bash
pip install numpy torch scikit-learn matplotlib jupyter
jupyter notebook example.ipynb
```

The notebook runs in about ten minutes on a GPU (60+80-epoch demo schedule, reaching ~0.99 balanced accuracy on MFPT); the paper configuration uses 120+160 epochs and five seeds. On CPU, reduce the epochs in the config cell for a quick pass.
