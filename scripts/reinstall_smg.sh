#!/bin/bash

source /jizhicfs/johnnyslin/env/psrl.sh
pip uninstall -y smg
pushd /jizhicfs/johnnyslin/psrl_router/workspace/psrl/third_party/smg
cargo build --release
cd bindings/python
maturin develop
popd
