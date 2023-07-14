import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from utils import Encoder
from datasets import RoboEireanDataModule
from models import JetNet, SingleShotDetector, ObjectDetectionTask


if __name__ == "__main__":
    LEARNING_RATE = 1e-3
    ALPHA = 2.0
    NUM_CLASSES = 1
    DEFAULT_SCALINGS = torch.tensor(
        [
            [0.06539708, 0.1283451 ],
       [0.119685  , 0.26551536],
       [0.20956908, 0.3853869 ],
       [0.3152952 , 0.4801411 ],
       [0.41862726, 0.8178706 ],
       [0.7547574 , 0.8290733 ]
        ]
    )
    encoder = Encoder(DEFAULT_SCALINGS, NUM_CLASSES)
    model = JetNet(NUM_CLASSES, DEFAULT_SCALINGS.shape[0])
    loss = SingleShotDetector(ALPHA)
    data_module = RoboEireanDataModule("data/raw/", encoder, 128)
    data_module.setup("fit")

    CHECKPOINT = True

    if CHECKPOINT:

        VERSION = 98

        checkpoint = os.listdir(f"new_logs/lightning_logs/version_{VERSION}/checkpoints")[0]
        checkpoint_path = f"new_logs/lightning_logs/version_{VERSION}/checkpoints/{checkpoint}"
        print(checkpoint_path)

        task = ObjectDetectionTask.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            loss=loss,
            encoder=encoder,
            learning_rate=LEARNING_RATE)

    else:
        task = ObjectDetectionTask(model, loss, encoder, LEARNING_RATE)

    logger = TensorBoardLogger(save_dir="new_logs")
    trainer = pl.Trainer(max_epochs=200, logger=logger)
    trainer.fit(model=task, datamodule=data_module)
