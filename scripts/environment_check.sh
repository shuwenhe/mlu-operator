#!/usr/bin/env bash
set -eu
uname -sr
getconf GNU_LIBC_VERSION
cat /sys/module/cambricon_drv/version 2>/dev/null || true
printf 'SDK prefix: %s\n' "${NEUWARE_ROOT:-/usr/local/neuware}"
for tool in cc cmake cncc cnmon; do command -v "$tool" || true; done
find /dev -maxdepth 1 -name 'cambricon*' -ls
for path in include/cnrt.h include/cnnl.h lib64/libcnrt.so lib64/libcnnl.so lib64/libcncl.so; do
    test ! -e "${NEUWARE_ROOT:-/usr/local/neuware}/$path" || ls -l "${NEUWARE_ROOT:-/usr/local/neuware}/$path"
done
