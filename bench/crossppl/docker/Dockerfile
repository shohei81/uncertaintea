# Fair-comparison container for the CPU legs of the cross-PPL benchmark
# (issue #121): every CPU framework (UncertainTea, NumPyro, CmdStan) runs in
# this single pinned Linux environment.  Metal runs stay native on the host —
# the container cannot reach the GPU — and are labelled as such in the
# results.
#
# Build (from bench/crossppl):
#   docker build -t uncertaintea-crossppl -f docker/Dockerfile .
# Run (repo mounted read-write for results/; keep a named volume for the
# Julia depot so instantiation happens once):
#   docker run --rm -v "$(git rev-parse --show-toplevel)":/work \
#       -v uncertaintea-crossppl-depot:/opt/julia-depot \
#       -w /work/bench/crossppl uncertaintea-crossppl ./run_all.sh cpu

FROM julia:1.12.2-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

ENV JULIA_DEPOT_PATH=/opt/julia-depot \
    CROSSPPL_CMDSTAN_DIR=/opt/cmdstan \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    CROSSPPL_IN_CONTAINER=1

# Pre-bake the pinned Python environment and CmdStan so container runs are
# reproducible and start fast.
COPY python/pyproject.toml python/uv.lock /opt/bench-python/
RUN cd /opt/bench-python && uv sync --frozen \
    && uv run --project /opt/bench-python --no-sync python -c \
        "import cmdstanpy; cmdstanpy.install_cmdstan(dir='/opt/cmdstan', version='2.36.0')"

WORKDIR /work/bench/crossppl
