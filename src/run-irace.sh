#!/usr/bin/env bash
set -ex

chkp_mode=$1
shift 1

dir=$1
shift 1

uuid=$(uuidgen)
dt=$(date '+%Y-%m-%dT%H:%M:%S')

log_file="/var/lib/irace/irace-log-${dt}-${uuid}.Rdata"
recovery_file="/var/lib/irace/irace-backup.Rdata"

case "${chkp_mode}" in
"nochkp")
    echo "Using no checkpoint as 'nochkp' mode is specified."
    args="--log-file ${log_file}"
    ;;

"usechkp")
    echo "Trying to find a checkpoint as 'usechkp' mode is specified."
    /app/make-recovery-file.py /var/lib/irace/
    recovery_file="/var/lib/irace/recovery_rdata.checkpoint"
    args="--recovery-file ${recovery_file} --log-file ${log_file}"

    if [[ -f "${recovery_file}" ]]; then
      echo "Found recovery file at ${recovery_file}. Using it."
      args="--recovery-file ${recovery_file} --log-file ${log_file}"
    else
      echo "No recovery file found at ${recovery_file}. Using it."
      # we can continue without recovery file, but it may be undesirable for a user
      # so better to signal it by failing
      exit 1
#      args="--log-file ${log_file}"
    fi

    ;;

*)
    echo "Unknown checkpoint mode: ${chkp_mode}. Only the following is supported: [nochkp, usechkp]"
    ;;

esac

cd "${dir}" || exit 1
exec "$IRACE_HOME"/bin/irace ${args} "${@}"
