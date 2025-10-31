from typing import Optional, Union

import lightning as pl
import torch
from hexray25.registries import CallbackRegistry, DataRegistry, ModelRegistry
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from pydantic import BaseModel

pl.seed_everything(42)


class BaseTrainer(BaseModel):
    model: Optional[pl.LightningModule] = None
    trainer: Optional[pl.Trainer] = None
    data_module: Optional[pl.LightningDataModule] = None
    training_params: Optional[Union[dict, BaseModel]] = None
    wandb_logger: Optional[WandbLogger] = None

    class Config:
        arbitrary_types_allowed = True

    def _setup_callbacks(self):
        """Set up callbacks from config"""
        callbacks = []
        for callback_config in self.training_params.callbacks:
            try:
                callback = CallbackRegistry.get(
                    callback_config.name, **callback_config.params
                )
                callbacks.append(callback)
                logger.debug(
                    f"Added callback {callback_config.name} with params {callback_config.params}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to setup callback {callback_config.name}: {str(e)}"
                )
                raise
        return callbacks

    def _setup_model(self):
        """Set up model from config"""
        try:
            model = ModelRegistry.get(
                self.training_params.model.name, **self.training_params.model.params
            )
            logger.info(f"Model {self.training_params.model.name} initialized")
            return model
        except Exception as e:
            logger.error(f"Failed to setup model: {str(e)}")
            raise

    def _setup_data(self):
        """Set up data from config"""
        try:
            data = DataRegistry.get(
                self.training_params.data.name, **self.training_params.data.params
            )
            logger.info(f"Data {self.training_params.data.name} initialized")
            return data
        except Exception as e:
            logger.error(f"Failed to setup data: {str(e)}")
            raise

    def _setup_wandb(self):
        """Set up wandb from config"""
        wandb_logger = WandbLogger(**self.training_params.wandb.params)
        logger.info("Wandb logger initialized")
        return wandb_logger

    def train(self):
        logger.info("Starting train()")
        try:
            self.model = self._setup_model()
            callbacks = self._setup_callbacks()
            self.data_module = self._setup_data()
            self.wandb_logger = self._setup_wandb()
            for torch_cfg in self.training_params.torch_config:
                if torch_cfg.name == "float32_matmul_precision":
                    torch.set_float32_matmul_precision(torch_cfg.value)

            self.trainer = pl.Trainer(
                **self.training_params.lightning.model_dump(
                    include={"lightning": {"params"}}
                ),
                callbacks=callbacks,
                logger=self.wandb_logger,
            )
            logger.debug(f"Initialized trainer with parameters: {self.training_params}")

            # Watch model after it's created
            self.wandb_logger.watch(self.model, log="all", log_freq=100)

            logger.info("Starting model training")
            self.trainer.fit(self.model, self.data_module)

            self.wandb_logger.experiment.unwatch(self.model)
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
