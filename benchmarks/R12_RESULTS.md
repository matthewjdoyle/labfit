# R12 optimization benchmark

Environment: Python 3.13.1, NumPy 2.2.6, SciPy 1.15.2, Windows-11-10.0.26200-SP0.

Configuration: `{'sizes': [32, 128, 512], 'grids': [200, 2000], 'repeats': 7, 'warmup': 2, 'seed': 20260930}`.

Reference: general dense solves on Cholesky factors and pointwise band derivatives. Full-fit comparisons replace only the whitening solver; patch setup is outside the timer. Both variants include the same covariance validation, factorization and optimization.

All numerical comparisons passed (rtol=2e-6, atol=1e-9). Raw samples, exact source hashes, git status and requested thread limits are recorded in results.json and samples.csv.

| Operation | Model | Points | Reference ms | Optimized ms | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| whitening | shared | 32 | 0.0073 | 0.0070 | 1.04x |
| correlated_fit | linear | 32 | 0.5789 | 0.5473 | 1.06x |
| correlated_fit | gaussian | 32 | 0.8332 | 0.8015 | 1.04x |
| whitening | shared | 128 | 0.0689 | 0.0098 | 7.03x |
| correlated_fit | linear | 128 | 1.5887 | 1.1184 | 1.42x |
| correlated_fit | gaussian | 128 | 2.4488 | 1.3603 | 1.80x |
| whitening | shared | 512 | 2.9131 | 0.0621 | 46.91x |
| correlated_fit | linear | 512 | 68.8738 | 39.1914 | 1.76x |
| correlated_fit | gaussian | 512 | 99.4199 | 41.6655 | 2.39x |
| confidence_band | linear | 200 | 3.8899 | 0.1159 | 33.56x |
| prediction_band | linear | 200 | 3.6059 | 0.1295 | 27.84x |
| confidence_band | linear | 2000 | 34.5065 | 0.2030 | 169.98x |
| prediction_band | linear | 2000 | 35.1737 | 0.2585 | 136.07x |
| confidence_band | gaussian | 200 | 8.4800 | 0.1503 | 56.42x |
| prediction_band | gaussian | 200 | 8.5916 | 0.1984 | 43.30x |
| confidence_band | gaussian | 2000 | 76.6417 | 0.3341 | 229.40x |
| prediction_band | gaussian | 2000 | 78.0531 | 0.4058 | 192.34x |

Times are medians from this machine. Band workloads time computation only; they do not include Matplotlib rendering or file output. Whitening excludes the one-time Cholesky factorization. Timings vary with hardware, BLAS and thread settings.

Recorded 2026-09-30. Raw artifacts for this run: [JSON report](results/r12/20260930T131712874562Z/results.json), [timing samples](results/r12/20260930T131712874562Z/samples.csv). These generated files are kept locally under the ignored results directory.

Maximum absolute numerical differences across all benchmark sizes:

| Operation | Metric | Maximum difference |
| --- | --- | ---: |
| whitening | values | 1.78e-15 |
| correlated_fit | parameters | 3.46e-10 |
| correlated_fit | covariance | 1.87e-12 |
| correlated_fit | fitted_values | 3.62e-10 |
| correlated_fit | statistics | 1.53e-13 |
| confidence_band | values | 8.88e-16 |
| prediction_band | values | 8.88e-16 |

Source SHA-256 hashes identify the measured working-tree implementation:

- `labfit/fitter_impl.py`: `a8039da7273ce783eb1bc3347085c4eb522b564a3eec4a9d477b749bd9ec00a7`
- `labfit/plot.py`: `291b00e3163b23b0d80977ff839baa7ae6ec4bb23c924137ce048acfdb75efa5`
- `benchmarks/benchmark_optimizations.py`: `b96e3da0f67f5465e1fba7e7281acbc222035d795d5ec2946053d8fe2adf7836`
