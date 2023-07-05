from utils import Encoder
import torch
from datasets import RoboEireanDataModule
import lightning.pytorch as pl

from models import JetNet, SingleShotDetectorLoss, ObjectDetectionTask
from lightning.pytorch.loggers import TensorBoardLogger
torch.manual_seed(0)

if __name__ == "__main__":
    LEARNING_RATE = 0.01
    ALPHA = 2.0
    NUM_CLASSES = 1
    DEFAULT_SCALINGS = torch.tensor(
        [
            [0.0788409 , 0.08730039],
            [0.12153658, 0.20424528],
            [0.2331543 , 0.25296658],
            [0.36146814, 0.49899086],
            [0.39320916, 0.9054488 ],
            [0.97875   , 0.9608333 ],
        ]
    )
    encoder = Encoder(DEFAULT_SCALINGS, NUM_CLASSES)
    model = JetNet(NUM_CLASSES, DEFAULT_SCALINGS.shape[0])
    loss = SingleShotDetectorLoss(ALPHA)
    data_module = RoboEireanDataModule("data/coco_nao/", encoder, 128)
    data_module.setup("fit")
    task = ObjectDetectionTask(model, loss, encoder, LEARNING_RATE)
    logger = TensorBoardLogger(save_dir="new_logs")
    trainer = pl.Trainer(max_epochs=10, logger=logger)
    trainer.fit(model=task, datamodule=data_module)
