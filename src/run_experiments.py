import argparse
import uuid
import yaml
from compare_experiments import compare
from src.config import experiment_config, load_config, project_path
from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    with project_path(args.experiments).open(encoding="utf-8") as stream:
        experiments = yaml.safe_load(stream)["experiments"]
    if len(experiments) < 5:
        raise ValueError("At least five experiments are required")
    batch_id = uuid.uuid4().hex
    for experiment in experiments:
        train(experiment_config(cfg, experiment), experiment["name"], batch_id)
    compare(cfg, batch_id=batch_id)


if __name__ == "__main__":
    main()
