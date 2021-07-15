#!/usr/bin/env bash

set -ex

if [[ -z "${KUBE_NAMESPACE}" ]]
then
  kubectl_args=""
else
  kubectl_args="-n ${KUBE_NAMESPACE}"
fi

exec kubectl ${kubectl_args} delete -f conf/repeater-job.yaml --ignore-not-found=true