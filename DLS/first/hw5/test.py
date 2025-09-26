import pickle

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# from models.resnet import load_resnet34
from models.meduim import MediumCnn
from models.abstract import BASE_DIR
from libs.dataset import SimpsonsDataset
from libs.predict import predict
from settings import TEST_DIR, DATA_DIR

test_files = sorted(list(TEST_DIR.rglob("*.jpg")))
test_dataset = SimpsonsDataset(test_files, mode="test")
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)
# model = load_resnet34(42, BASE_DIR.joinpath("Medium/10-11-2024_12-01/model"))
model = MediumCnn.load(42, BASE_DIR.joinpath("Medium/10-11-2024_12-01/model"))
probs = predict(model, test_loader)


label_encoder = pickle.load(open(SimpsonsDataset.LABEL_DIR, "rb"))

preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
test_filenames = [path.name for path in test_dataset.files]

my_submit = pd.DataFrame({"Id": test_filenames, "Expected": preds})
print(my_submit.head())

my_submit.to_csv(DATA_DIR.joinpath("еуые.csv"), index=False)
