#!/usr/bin/env bash

set -ex

if [[ -z "${KUBE_NAMESPACE}" ]]
then
  kubectl_args=""
else
  kubectl_args="-n ${KUBE_NAMESPACE}"
fi

exec kubectl "${kubectl_args}" apply -f conf/irace-runner-job.yaml
