from typing import Dict, Any


def make_log_config_dict(filename: str = "/var/log/tm-alg.txt") -> Dict[str, Any]:
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
                'filename': filename,
            }
        },
        'loggers': {
            'GA': {
                'handlers': ['default', 'logfile'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }
