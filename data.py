from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def download_MNIST(save_dir):
    try:
        # First try the dataset name approach
        x, y = fetch_openml("mnist_784", version=1, data_home="data_cache", return_X_y=True)
    except Exception:
        # If that fails, try using the numeric ID approach
        try:
            x, y = fetch_openml(data_id=554, version=1, data_home="data_cache", return_X_y=True)
        except Exception:
            # If both fail, use the more modern approach with newer dataset ID
            x, y = fetch_openml("mnist_784", version=1, parser="auto", data_home="data_cache", return_X_y=True, as_frame=False)

    # Normalize the data
    x = x.astype(np.float32)
    x /= 255.0
    x -= x.mean()
    
    # Convert y to DataFrame for one-hot encoding
    y_df = pd.DataFrame({'digit': y})
    y = pd.get_dummies(y_df['digit']).to_numpy().astype(np.float32)

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.15, random_state=42
    )
    
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train data
    pd.DataFrame(x_train).to_parquet(save_dir / "x_train.parquet")
    pd.DataFrame(x_val).to_parquet(save_dir / "x_val.parquet")
    np.save(save_dir / "y_train.npy", y_train)
    np.save(save_dir / "y_val.npy", y_val)


if __name__ == "__main__":
    save_dir = Path("data/mnist_784/")
    print(f"Downloading MNIST dataset at {save_dir.resolve()}")
    download_MNIST(save_dir)
