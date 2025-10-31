from hexray25.configs.vjt_bg_train import vjt_bg
from hexray25.runners.base_trainer import BaseTrainer

# Initialize the trainer with the default configuration
trainer = BaseTrainer(training_params=vjt_bg)

# Start training
trainer.train()