#%%
# Import the required modules
from anomalib.data import MVTec, Visa
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Initialize the datamodule, model and engine
# datamodule_mvtec = MVTec()
datamodule_visa = Visa()
# datamodule_led = Folder()

#%%
model = Patchcore()
engine = Engine()

# Train the model
engine.fit(datamodule=datamodule_visa, model=model)

#%%