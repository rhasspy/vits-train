#!/usr/bin/env bash
set -e

if [[ -z "${PIP_INSTALL}" ]]; then
    PIP_INSTALL='install'
fi

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"
src_dir="$(realpath "${this_dir}/..")"

# -----------------------------------------------------------------------------

venv="${src_dir}/.venv"

# -----------------------------------------------------------------------------

: "${PYTHON=python3}"

python_version="$(${PYTHON} --version)"

# Create virtual environment
echo "Creating virtual environment at ${venv} (${python_version})"
rm -rf "${venv}"
"${PYTHON}" -m venv "${venv}"
source "${venv}/bin/activate"

# Install Python dependencies
echo 'Installing Python dependencies'
pip3 ${PIP_INSTALL} --upgrade pip

# Install vits_train
pushd "${src_dir}"
pip3 ${PIP_INSTALL} .[dev]
popd

pushd "${src_dir}/vits_train/monotonic_align"
echo 'Compiling monotonic_align extension'
python3 setup.py build_ext --inplace
popd

# -----------------------------------------------------------------------------

echo "OK"
