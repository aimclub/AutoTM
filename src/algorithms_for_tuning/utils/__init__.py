import os
from typing import Dict, Any, Optional


def make_log_config_dict(filename: str = "/var/log/tm-alg.txt", uid: Optional[str] = None) -> Dict[str, Any]:
    if uid:
        dirname = os.path.dirname(filename)
        file, ext = os.path.splitext(os.path.basename(filename))
        log_filename = os.path.join(dirname, f"{file}-{uid}.{ext}")
    else:
        log_filename = filename
    return {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'ERROR',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': log_filename,
            }
        },
        'loggers': {
            'root': {
                'handlers': ['default', 'logfile'],
                'level': 'DEBUG',
                'propagate': False
            },
            'GA': {
                'handlers': ['default', 'logfile'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
