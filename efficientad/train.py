import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.models import get_model
from anomalib.data import get_datamodule
from pytorch_lightning.loggers import TensorBoardLogger

# Define your configuration directly in the script (you can use a dictionary to mimic the YAML config)
config = {
    "model": {
        "name": "EfficientAD",
        "input_size": 224,
        "backbone": "efficientnet_b0",
        "pretrained": True
    },
    "dataset": {
        "name": "CustomDataset",
        "format": "folder",
        "path": "/path/to/custom_dataset",
        "image_size": (224, 224),
    },
    "train": {
        "batch_size": 32,
        "num_epochs": 100,
        "shuffle": True,
        "num_workers": 4,
        "early_stopping_patience": 10
    },
    "optimization": {
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "weight_decay": 0.0001,
    },
    "logging": {
        "logger": "tensorboard",
        "log_dir": "logs/"
    }
}

# Create model and data module using the configuration
model = get_model(config)
datamodule = get_datamodule(config)

# Create logger for TensorBoard
logger = TensorBoardLogger(save_dir=config["logging"]["log_dir"], name="EfficientAD")

# Define callbacks
checkpoint_callback = ModelCheckpoint(monitor="val_auc", mode="max", save_top_k=1)
early_stopping_callback = EarlyStopping(monitor="val_auc", patience=config["train"]["early_stopping_patience"], mode="max")

# Initialize the Trainer with the callbacks
trainer = Trainer(
    max_epochs=config["train"]["num_epochs"],
    logger=logger,
    callbacks=[checkpoint_callback, early_stopping_callback],
    accelerator="auto"  # This will use GPU if available, otherwise fallback to CPU
)

# Train the model
trainer.fit(model, datamodule=datamodule)

# Test the model
trainer.test(model, datamodule=datamodule)

# Now for inference, you can load the trained model and run inference on new images
from anomalib.deploy import Inferencer

# Specify the path to the best model checkpoint
checkpoint_path = checkpoint_callback.best_model_path

# Create the inferencer with the trained model
inferencer = Inferencer(config=config, model_path=checkpoint_path)

# Run inference on a new image
output = inferencer.predict("path/to/new_image.png")

# Print the anomaly score
print(f"Anomaly Score: {output['anomaly_score']}")
