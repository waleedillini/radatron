import logging
from detectron2.config import *
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import RotatedCOCOEvaluator,DatasetEvaluators
from detectron2.engine import default_argument_parser, launch
import os
import torch
from radatron.modeling import *
from radatron.data import RadatronMapper
from radatron.evaluation import RadatronEvaluator
from radatron.config.radatron_setup import setup
from radatron.data.dataset.register_radatron_dataset import register_radatron

logger = logging.getLogger("detectron2")
    
class RadatronTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RadatronEvaluator(dataset_name, cfg, False, output_folder)]
        #evaluators = [RotatedCOCOEvaluator(dataset_name, cfg, False, output_folder)]
        return DatasetEvaluators(evaluators)
      
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=RadatronMapper(cfg, mode='train'))
    
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=RadatronMapper(cfg, mode='eval'))
    

def main(args):
    
    cfg = setup(args)

    register_radatron(cfg)
    trainer = RadatronTrainer(cfg)

    with torch.autograd.set_detect_anomaly(True):
        trainer.resume_or_load(resume=args.resume)
        return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
