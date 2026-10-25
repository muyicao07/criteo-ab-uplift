#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  PROJECT_PYTHON="${PYTHON_BIN:-}"
  if [[ -z "$PROJECT_PYTHON" ]]; then
    for candidate in python3.12 python3 python /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12; do
      if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info[:2] == (3,12)' >/dev/null 2>&1; then
        PROJECT_PYTHON="$(command -v "$candidate")"
        break
      fi
    done
  fi
  if [[ -z "$PROJECT_PYTHON" ]]; then
    echo 'Python 3.12 is required for the pinned, reproducible environment. Install Python 3.12, then run this same command again.'
    echo 'You can also select an installed interpreter with PYTHON_BIN=/path/to/python3 bash run.sh'
    exit 1
  fi
  if ! "$PROJECT_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12)' >/dev/null 2>&1; then
    echo "The selected interpreter is not Python 3.12: $PROJECT_PYTHON"
    exit 1
  fi
  echo 'Creating the project Python 3.12 environment...'
  "$PROJECT_PYTHON" -m venv .venv
fi

PROJECT_PYTHON="$PROJECT_DIR/.venv/bin/python"
if ! "$PROJECT_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12)' >/dev/null 2>&1; then
  echo 'The existing .venv uses a different Python version. Rename that folder, then run again with Python 3.12.'
  exit 1
fi
DEP_SIGNATURE="$("$PROJECT_PYTHON" -c 'import hashlib,pathlib;print(hashlib.sha256(pathlib.Path("pyproject.toml").read_bytes()+pathlib.Path("requirements-lock.txt").read_bytes()).hexdigest())')"
if [[ ! -f .venv/.criteo-dependencies ]] || [[ "$(cat .venv/.criteo-dependencies)" != "$DEP_SIGNATURE" ]]; then
  echo 'Installing the pinned dependencies (automatically skipped on later runs)...'
  "$PROJECT_PYTHON" -m pip install -r requirements-lock.txt
  "$PROJECT_PYTHON" -m pip install --no-deps -e .
  printf '%s\n' "$DEP_SIGNATURE" > .venv/.criteo-dependencies
fi

# Reuse the OpenMP runtime shipped in the installed scikit-learn macOS wheel.
# This does not change system libraries or require Homebrew installation.
if [[ "$(uname -s)" == Darwin ]]; then
  # Resolve .venv symlinks before building a colon-separated loader search path.
  OMP_DIR="$("$PROJECT_PYTHON" -c 'import pathlib,importlib.util; p=(pathlib.Path(importlib.util.find_spec("sklearn").origin).parent/".dylibs").resolve(); print(p if (p/"libomp.dylib").is_file() else "")')"
  if [[ -n "$OMP_DIR" ]]; then
    export DYLD_LIBRARY_PATH="$OMP_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
  fi
fi
export OMP_NUM_THREADS="${CRITEO_THREADS:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export VECLIB_MAXIMUM_THREADS="$OMP_NUM_THREADS"
export LOKY_MAX_CPU_COUNT="$OMP_NUM_THREADS"
export MPLCONFIGDIR="$PROJECT_DIR/results/.mpl-cache"
export JUPYTER_RUNTIME_DIR="$PROJECT_DIR/results/.jupyter"
mkdir -p "$MPLCONFIGDIR" "$JUPYTER_RUNTIME_DIR"
echo 'Running the complete full-data Criteo project...'
"$PROJECT_PYTHON" -m criteo_ab_uplift run --threads "$OMP_NUM_THREADS" "$@"
