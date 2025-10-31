import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from hexray25.configs.vjt_v32_all_defect import vjt_v32_all_defect
from hexray25.runners.base_trainer import BaseTrainer

# Initialize the trainer with the default configuration
trainer = BaseTrainer(training_params=vjt_v32_all_defect)

# Start training
trainer.train()