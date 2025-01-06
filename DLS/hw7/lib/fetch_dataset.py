import os

import numpy as np

import pandas as pd
import skimage.io
from skimage.transform import resize


def fetch_dataset(
    attrs_name="lfw_attributes.txt",
    images_name="lfw-deepfunneled",
    dx=80,
    dy=80,
    dimx=64,
    dimy=64,
):
    # read attrs
    df_attrs = pd.read_csv(
        attrs_name,
        sep="\t",
        skiprows=1,
    )
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])

    # read photos
    photo_ids = []
    for dirpath, dirnames, filenames in os.walk(images_name):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath, fname)
                photo_id = fname[:-4].replace("_", " ").split()
                person_id = " ".join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append(
                    {"person": person_id, "imagenum": photo_number, "photo_path": fpath}
                )

    photo_ids = pd.DataFrame(photo_ids)
    df = pd.merge(df_attrs, photo_ids, on=("person", "imagenum"))

    assert len(df) == len(df_attrs), "lost some data when merging dataframes"

    all_photos = (
        df["photo_path"]
        .apply(skimage.io.imread)
        .apply(lambda img: img[dy:-dy, dx:-dx])
        .apply(lambda img: resize(img, [dimx, dimy]))
    )

    all_photos = np.stack(all_photos.values)  # .astype('uint8')
    all_attrs = df.drop(["photo_path", "person", "imagenum"], axis=1)

    return all_photos, all_attrs
