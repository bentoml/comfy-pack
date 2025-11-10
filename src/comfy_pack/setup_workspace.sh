#!/bin/bash

set -eo pipefail

CPACK=/tmp/bento-cpack
mkdir -p "$CPACK"

cat <<EOF | head -c -1 > "$CPACK"/snapshot.json
<SNAPSHOT>
EOF

checksum=$(md5sum "$CPACK"/snapshot.json | awk '{print $1}')
workspace="${BENTOML_HOME:-$HOME/bentoml}/run/comfy_workspace/${checksum}"

if [ -n "$VIRTUAL_ENV" ]; then
    # shellcheck disable=SC1091
    source "$VIRTUAL_ENV"/bin/activate
fi

set -x
comfy-pack unpack "$CPACK" -d "$workspace" --no-models --no-venv -v
chown -R bentoml:bentoml "$workspace"
set +x
