#!/usr/bin/env bash

set -ex

if [[ -z "${KUBE_NAMESPACE}" ]]
then
  kubectl_args=""
else
  kubectl_args="-n ${KUBE_NAMESPACE}"
fi

# check for presence of the first argument
[[ $# -eq 1 ]] || exit 1
alg_name=$1

exec kubectl ${kubectl_args} apply -f conf/irace-runner-job-"${alg_name}".yaml
