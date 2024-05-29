import argparse
import logging
import os

from jinja2 import Environment, FileSystemLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description='Generate deploy config with'
                                                 ' entities for distributed fitness estimation')

    parser.add_argument("--host-data-dir",
                        type=str,
                        required=True,
                        help="Path to a directory on hosts to be mounted in containers of workers")

    parser.add_argument("--datasets-config",
                        type=str,
                        required=True,
                        help="Path to a directory on hosts to be mounted in containers of workers")

    parser.add_argument("--host-data-dir-mount-path", type=str, default="/storage",
                        help="Path inside containers of workers where to mount host_data_dir")

    parser.add_argument("--registry", type=str, default=None,
                        help="Registry to push images to")

    parser.add_argument("--client_image", type=str, default="fitness-client:latest",
                        help="Image to use for a client pod")

    parser.add_argument("--flower_image", type=str, default="flower:latest",
                        help="Image to use for Flower - a celery monitoring tool")

    parser.add_argument("--worker_image", type=str, default="fitness-worker:latest",
                        help="Image to use for worker pods")

    parser.add_argument("--worker_count", type=int, default=1,
                        help="Count of workers to be launched on the kubernetes cluster")

    parser.add_argument("--worker_cpu", type=int, default=1,
                        help="Count of cpu (int) to be allocated per worker")

    parser.add_argument("--worker_mem", type=str, default="1G",
                        help="Amount of memory to be allocated per worker")

    parser.add_argument("--config_template_dir", default="deploy",
                        help="Path to a template's dir the config will be generated from")

    parser.add_argument("--out_dir", type=str, default=None,
                        help="Path to a generated config")

    parser.add_argument("--mongo_collection", type=str, default='main_tm_stats',
                        help="Mongo collection")

    args = parser.parse_args()

    worker_template_path = "kube-fitness-workers.yaml.j2"
    client_template_path = "kube-fitness-client-job.yaml.j2"

    if args.out_dir:
        worker_cfg_out_path = os.path.join(args.out_dir, "kube-fitness-workers.yaml")
        client_cfg_out_path = os.path.join(args.out_dir, "kube-fitness-client-job.yaml")
    else:
        worker_cfg_out_path = os.path.join(args.config_template_dir, "kube-fitness-workers.yaml")
        client_cfg_out_path = os.path.join(args.config_template_dir, "kube-fitness-client-job.yaml")

    datasets_config = args.datasets_config \
        if os.path.isabs(args.datasets_config) else os.path.join(args.config_template_dir, args.datasets_config)

    logging.info(f"Reading datasets config from {args.datasets_config}")
    with open(datasets_config, "r") as f:
        datasets_config_content = f.read()

    logging.info(f"Using template dir: {args.config_template_dir}")
    logging.info(f"Using template {worker_template_path}")
    logging.info(f"Generating config file {worker_cfg_out_path}")

    env = Environment(loader=FileSystemLoader(args.config_template_dir))
    template = env.get_template(worker_template_path)
    template.stream(
        flower_image=f"{args.registry}/{args.flower_image}" if args.registry else args.flower_image,
        image=f"{args.registry}/{args.worker_image}" if args.registry else args.worker_image,
        pull_policy="Always" if args.registry else "IfNotPresent",
        worker_count=args.worker_count,
        worker_cpu=args.worker_cpu,
        worker_mem=args.worker_mem,
        host_data_dir=args.host_data_dir,
        host_data_dir_mount_path=args.host_data_dir_mount_path,
        datasets_config_content=datasets_config_content,
        mongo_collection=args.mongo_collection
    ).dump(worker_cfg_out_path)

    template = env.get_template(client_template_path)
    template.stream(
        image=f"{args.registry}/{args.client_image}" if args.registry else args.client_image,
        pull_policy="Always" if args.registry else "IfNotPresent",
    ).dump(client_cfg_out_path)


if __name__ == "__main__":
    main()
