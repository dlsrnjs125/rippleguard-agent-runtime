# Troubleshooting

## XGBoost import fails on macOS

현상: `libxgboost.dylib` cannot load `libomp.dylib`.

영향 범위: local training, integration tests and readiness.

재현 방법: run `python3 -c "import xgboost"`.

원인: XGBoost macOS wheel requires the OpenMP runtime.

선택한 해결책: install `libomp` locally with Homebrew.

검증 방법: `make train-baseline` and `make integration-test`.

후속 Repository 영향: none.
