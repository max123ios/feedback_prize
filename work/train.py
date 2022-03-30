from venv import create
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from dataloader import create_dataloader
from model import create_model
from omegaconf import DictConfig
import hydra
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
seed_everything(42)

def train(cfg:DictConfig):
    train_data, val_data, train_label_csv, val_label_csv = create_dataloader(**cfg)
    model = create_model(cfg, train_label_csv, val_label_csv)
    # # test_model_ckpt = '/root/projects/feedback_prize/work/outputs/2022-03-07/bestscoredivil-ennergy/outputs/feedback/33c9a0w2/checkpoints/epoch=5-val_f1_score=0.641-val_loss=0.636.ckpt'
    # # checkpoint = torch.load(test_model_ckpt)
    # # model.load_state_dict(checkpoint['state_dict'],False)
    wandb_logger = WandbLogger(entity = 'wangm', project='feedback')
    trainer = Trainer(gpus=cfg.gpus,
                            max_epochs=cfg.num_epochs,
                            # check_val_every_n_epoch = 2,
                            # persistent_workers=True,
                            # strategy='ddp',
                            default_root_dir = './outputs/',
                            precision=16,
                            logger = wandb_logger,
                            num_sanity_val_steps = 0,
                            callbacks=[
                                ModelCheckpoint(monitor='val_f1_score', mode='max', save_top_k=3, filename="{epoch:d}-{val_f1_score:.3f}-{val_loss:.3f}")
                            ]
                            )
    
    trainer.fit(model, train_dataloaders = train_data, val_dataloaders = val_data)


@hydra.main(config_path="conf", config_name="config")
def main(cfg:DictConfig):
    train(cfg)


if __name__ == '__main__':
    main()