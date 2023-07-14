from utils import Encoder
import torch
from datasets import RoboEireanDataModule
import lightning.pytorch as pl

from models import JetNet, SingleShotDetector, ObjectDetectionTask
from lightning.pytorch.loggers import TensorBoardLogger

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
    task = ObjectDetectionTask(model, loss, encoder, LEARNING_RATE)
    
    # view logs with tensorboard --logdir new_logs
    
    logger = TensorBoardLogger(save_dir="new_logs")
    trainer = pl.Trainer(max_epochs=200, logger=logger)
    trainer.fit(model=task, datamodule=data_module)