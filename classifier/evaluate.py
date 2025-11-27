from ultralytics import YOLO
import yaml
from argparse import Namespace

if __name__ == "__main__":

    config_path = "./config.yaml"

    print(f"Loading config from: {config_path}")
    with open(config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    
    model = YOLO(conf.model_path)

    metrics = model.val(data='config.yaml', split='val')

    print("Dataset validation complete, results are in runs folder.")