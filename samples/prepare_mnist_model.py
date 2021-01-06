from automation.create_mnist_model import create
from data.model_data import ModelTrainType

create(name="showcase_v3_regu", batch_size=128, epochs=15, layer_data=[81, 49], regularized=True)
