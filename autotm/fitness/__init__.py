import os

# local or cluster
SUPPORTED_EXEC_MODES = ['local', 'cluster']
AUTOTM_EXEC_MODE = os.environ.get("AUTOTM_EXEC_MODE", "local")


# head or worker
SUPPORTED_COMPONENTS = ['head', 'worker']
AUTOTM_COMPONENT = os.environ.get("AUTOTM_COMPONENT", "head")
