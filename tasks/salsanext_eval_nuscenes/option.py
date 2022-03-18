import os 
import yaml
import sys 
import shutil

sys.path.insert(0, "../../../")

import pc_processor

class Option(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, "r"))

        # ---------------------------- general options -----------------
        self.save_path = self.config["pretrained_path"]# log path
        self.seed = self.config["seed"] # manually set RNG seed
        self.gpu = self.config["gpu"] # GPU id to use, e.g. "0,1,2,3"
        self.rank = 0 # rank of distributed thread
        self.world_size = 1 # 
        self.distributed = False # 
        self.n_gpus = len(self.gpu.split(",")) # # number of GPUs to use by default
        self.dist_backend = "nccl"
        self.dist_url = "env://"

        self.print_frequency = self.config["print_frequency"]  # print frequency (default: 10)
        self.n_threads = self.config["n_threads"] # number of threads used for data loading
        self.experiment_id = self.config["experiment_id"] # identifier for experiment
        self.is_debug = self.config["is_debug"]
       
        # --------------------------- data config ------------------------
        self.dataset = self.config["dataset"]
        self.n_classes = self.config["nclasses"]
        self.data_root = self.config["data_root"]
        self.has_label = self.config["has_label"]

        # --------------------------- model options -----------------------
        self.base_channels = self.config["base_channels"]
        self.img_backbone = self.config["img_backbone"]
        self.imagenet_pretrained = self.config["imagenet_pretrained"]

        # --------------------------- checkpoit model ----------------------
        self.pretrained_model = os.path.join(self.config["pretrained_path"], "checkpoint", self.config["best_model"])

        self._prepare()

    def _prepare(self):
        # ---- check params
        if self.net_type == "SalsaNextUncertainty":
            self.use_uncertainty = True

        # --------------------------------
        if not os.path.isdir(self.save_path):
            raise ValueError(f"training path not exists: {self.save_path}")

        if self.config["post"]["KNN"]["use"]:
            knn_str = f'KNN-{self.config["post"]["KNN"]["params"]["search"]}'
        else:
            knn_str = ""
        self.save_path = os.path.join(
            self.save_path,
            f"Eval-SV_{self.dataset}_{self.net_type}_{knn_str}_{self.experiment_id}",
        )

    def check_path(self): 
        if pc_processor.utils.is_main_process():
            if os.path.exists(self.save_path):
                print(f"file exist: {self.save_path}")
                action = input("Select Action: d(delete) / q(quit): ").lower().strip()
                if action == "d":
                    shutil.rmtree(self.save_path)
                else:
                    raise OSError(f"Directory exits: {self.save_path}")

            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
