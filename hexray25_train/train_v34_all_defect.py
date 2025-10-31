from hexray25.configs.vjt_v34_all_defect import vjt_v34_all_defect
from hexray25.runners.base_trainer import BaseTrainer

# Initialize the trainer with the default configuration
trainer = BaseTrainer(training_params=vjt_v34_all_defect)

# Start training
trainer.train()