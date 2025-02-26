import argparse

from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config/default.yaml")
args = parser.parse_args()

conf = OmegaConf.load(args.config)