#!/usr/bin/env bash
set -euo pipefail
cards=$1
shift
workspace=/app/shuwen/mlu-operator
out=$(cd "$(dirname "$0")" && pwd)
devices=(--device=/dev/cambricon_ctl:/dev/cambricon_ctl:rw)
for i in $cards; do
 for node in cambricon_dev"$i" cambricon_ipcm"$i"; do devices+=(--device="/dev/$node:/dev/$node:rw"); done
done
docker run --rm --pull=never --runtime=runc --user 1001:1001 --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777 "${devices[@]}" --mount "type=bind,src=$workspace,dst=/project,readonly" --mount "type=bind,src=$out,dst=/results" --workdir /project --entrypoint /bin/bash sha256:8e653daf38cb9cdd7a2120f9f7dcddcbe50677ac5c1c6e8d1243fa9576e26756 -ex -o pipefail -c "$1"
