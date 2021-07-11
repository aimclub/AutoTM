#!/usr/bin/env bash
set -ex

dir=$1

shift 1

uuid=$(uuidgen)
dt=$(date '+%Y-%m-%dT%H:%M:%S')

log_file="/var/lib/irace/irace-log-${dt}-${uuid}.Rdata"
recovery_file="/var/lib/irace/irace-backup.Rdata"

# TODO: fix functionality of checkpoint restoring
#if [[ -f "${log_file}" ]]; then
#    echo "${log_file} exists. Make it as a recovery file ${recovery_file}"
#    mv "${log_file}" "${recovery_file}"
#    args="--recovery-file ${recovery_file} --log-file ${log_file}"
#elif [[ -f "${recovery_file}" ]]; then
#    echo "No ${log_file} found. But recovery file ${recovery_file} exists. The run will continue from it."
#    args="--recovery-file ${recovery_file} --log-file ${log_file}"
#else
#    args="--log-file ${log_file}"
#fi

args="--log-file ${log_file}"

cd "${dir}" || exit 1
exec "$IRACE_HOME"/bin/irace ${args} "${@}"