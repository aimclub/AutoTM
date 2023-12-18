{{/*
Expand the name of the chart.
*/}}
{{- define "st-workspace.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "st-workspace.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "st-workspace.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "st-workspace.labels" -}}
helm.sh/chart: {{ include "st-workspace.chart" . }}
{{ include "st-workspace.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "st-workspace.selectorLabels" -}}
app.kubernetes.io/name: {{ include "st-workspace.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "st-workspace.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "st-workspace.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Helper function to define prefix for all entities
*/}}
{{- define "autotm.prefix" -}}
{{- $prefix := default .Values.autotm_prefix "" | trunc 16  -}}
{{- ternary $prefix (printf "%s-" $prefix) (empty $prefix) -}}
{{- end -}}

{{/*
Mlflow db url
*/}}
{{- define "autotm.mlflow_db_url" -}}
{{- if .Values.mlflow_enabled -}}
{{- printf "mysql+pymysql://%s:%s@%smlflow-db:3306/%s" (.Values.mlflow_mysql_user) (.Values.mlflow_mysql_password) (include "autotm.prefix" . ) (.Values.mlflow_mysql_database) -}}
{{- else -}}
""
{{- end -}}
{{- end -}}

{{/*
Mongo url
*/}}
{{- define "autotm.mongo_url" -}}
{{- if .Values.mongo_enabled -}}
{{- printf "mongodb://%s:%s@%smongo-tm-experiments-db:27017" (.Values.mongo_user) (.Values.mongo_password) (include "autotm.prefix" .) -}}
{{- else -}}
""
{{- end -}}
{{- end -}}

{{/*
Celery broker url
*/}}
{{- define "autotm.celery_broker_url" -}}
{{- printf "amqp://guest:guest@%srabbitmq-service:5672" (include "autotm.prefix" .) -}}
{{- end -}}

{{/*
Celery result backend url
*/}}
{{- define "autotm.celery_result_backend" -}}
{{- printf "redis://%sredis:6379/1" (include "autotm.prefix" .) -}}
{{- end -}}

{{/*
Check if required persistent volume exists.
One should invoke it like '{{ include "autotm.find_pv" (list . "pv_mlflow_db") }}', where 'pv_mlflow_db' is a variable from values.yaml
See:
- https://stackoverflow.com/questions/70803242/helm-pass-single-string-to-a-template
- https://austindewey.com/2021/06/02/writing-function-based-templates-in-helm/
*/}}
{{- define "autotm.find_pv" -}}
{{- $root := index . 0 -}}
{{- $pv_var := index . 1 -}}
{{- $pv_name := (printf "%s%s" (include "autotm.prefix" $root) (get $root.Values $pv_var)) -}}
{{- $result := (lookup "v1" "PersistentVolume" "" $pv_name) -}}
{{- if empty $result -}}
{{- (printf "Persistent volume with name '%s' not found" $pv_name) | fail -}}
{{- else -}}
{{- $result.metadata.name -}}
{{- end -}}
{{- end -}}
