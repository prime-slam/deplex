#!/bin/bash
# Copyright 2022 prime-slam
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
export LC_ALL=C
export MACOSX_DEPLOYMENT_TARGET=10.15

cmake -B build -DBUILD_TESTS=OFF -DBUILD_PYTHON=ON

for PYBIN in /Users/runner/hostedtoolcache/Python/3.*/x64/bin/python*?[0-9]
do
  cmake -B build -DPYTHON_EXECUTABLE="${PYBIN}"
  cmake --build build --target build-wheel
done

python3 -m pip install --user -q build
python3 -m build --wheel --outdir wheelhouse/ .