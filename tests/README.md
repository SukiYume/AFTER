# Test suite

The tests are organized by production module or workflow rather than by the
change that introduced them:

- `test_calibration*.py`: two/four-product calibration, noise-diode folding,
  provenance, downsampling, and output reuse.
- `test_cut_*.py`, `test_fits_to_h5.py`, and `test_batch_cut_burst_data.py`:
  raw-data selection, cut boundaries, legacy filenames, and H5 conversion.
- `test_burst_detect.py`, `test_burst_analysis.py`, `test_burst_properties.py`,
  `test_burst_dm.py`, `test_burst_pol.py`, and `test_rfi.py`: detection and
  single-burst analysis.
- `test_burst_sync_rm.py`: end-to-end synchronized-RM products and failure
  modes.
- `test_burst_dashboard.py`: user-visible dashboard output.

Run the full suite from the repository root:

```bash
python -m pytest -q
```

Inspect production-code coverage when changing behavior:

```bash
python -m pytest -q --cov=after --cov=batch_processing --cov-report=term-missing:skip-covered
```

Synthetic expectations should be independent of production constants and
private implementation details whenever a public output or error contract can
be asserted instead. Tests may patch plotting, GPU/model imports, or expensive
I/O, but should keep the scientific transformation under test real.
