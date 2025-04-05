import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

class Dataset:
    """
    Improved implementation of Dataset class that handles data loading and microbatching.
    Features:
    - Iterator-based microbatch loading
    - Lazy data loading with customizable loader function
    - Consistent microbatch handling across sequential and distributed training
    """
    def __init__(
        self, 
        save_dir:Path,
        global_batch_size: int, 
        mubatch_size:int , 
        validation:bool = False, 
        data_loader:Optional[Callable] = None, 

    ):
        """
        Initialize the dataset with configuration parameters.
        
        Args:
            save_dir: Directory containing the dataset files
            global_batch_size: Total batch size across all processes
            mubatch_size: Size of micro-batches
            validation: Whether to use validation data
            data_loader: Optional custom function to load data
        """
        assert save_dir.is_dir()

        self.save_dir = save_dir
        self.global_batch_size = global_batch_size
        self.mubatch_size = mubatch_size
        self.is_validation = validation 
        self.data_loader = data_loader

        self.input_X = None 
        self.target_y = None 
        self.local_batch_size = None
        self.DP_rank = None 
        self.DP_size = None 

    def load(self, DP_rank:int , DP_size:int) -> None:
        """
        Load the dataset for the current process based on data parallelism configuration.
        
        Args:
            DP_rank: Rank of the current process
            DP_size: Total number of data-parallel processes
        """
        assert DP_rank < DP_size, "Rank must be less than the total number of processes"
        assert self.global_batch_size % DP_size == 0, "Global batch size must be divisible by DP_size"
        assert (self.global_batch_size // DP_size) % self.mubatch_size == 0, "μBatch size must divide local batch size"

        self.DP_rank = DP_rank
        self.DP_size = DP_size
        self.local_batch_size = self.global_batch_size // DP_size
        
        if self.data_loader:
            self.input_X , self.target_y = self.data_loader(
                self.save_dir, self.is_validation , DP_rank, DP_size, self.global_batch_size
            )
        else:
            self._default_load()

         # Verify loaded data is properly sized
        assert len(self.input_X) % self.mubatch_size == 0
        assert len(self.input_X) % self.local_batch_size == 0
        assert len(self.input_X) == len(self.target_y)

        def _default_load(self) -> None:
         """Default data loading implementation"""
        suffix = "val" if self.is_validation else "train"
        
        # Load full dataset
        input_X = pd.read_parquet(self.save_dir / f"x_{suffix}.parquet").to_numpy(dtype=np.float32)
        target_y = np.load(self.save_dir / f"y_{suffix}.npy").astype(np.float32)
        
        assert len(input_X) == len(target_y), "Input and target lengths must match"
        
        # Drop samples to ensure even division by global batch size
        full_tiles_length = len(input_X) - (len(input_X) % self.global_batch_size)
        
        # Each process selects its subset with rank-offset and size-strides
        # Copy ensures contiguous memory layout for better performance
        self.input_X = input_X[self.DP_rank:full_tiles_length:self.DP_size].copy()
        self.target_y = target_y[self.DP_rank:full_tiles_length:self.DP_size].copy()

        def __len__(self) -> int:
            """Return the length of the dataset for this process"""
            if self.input_X is None:
                raise RuntimeError("Dataset must be loaded before length can be determined")
            return len(self.input_X)
    
        def get_num_batches(self) -> int:
            """Get the number of local batches for this process"""
            return len(self) // self.local_batch_size
    
        def get_num_mubatches(self) -> int:
            """Get the number of microbatches per local batch"""
            return self.local_batch_size // self.mubatch_size

        def batch_iterator(self, batch_id:int)-> Iterator[Tuple[np.ndarray, np.ndarray]]:
            """
            Create an iterator that yields microbatches for a given batch.
            
            Args:
                batch_id: The index of the batch
                
            Yields:
                Tuple of (input_X, target_y) for each microbatch
            """
            assert 0 <= batch_id < self.get_num_batchers(), "Invalid batch_id"

            start_idx = batch_id * self.local_btach_size

            for mubatch_id in range(self.get_num_mubatches()):
                mu_start = start_idx + mubatch_id * self.mubatch_size
                mu_end = mu_start + self.mubatch.size 

                yield self.input_X[mu_start:mu_end], self.target_y[mu_start:mu_end]

        def iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
            """
            Create an iterator that yields all microbatches in the dataset.
            
            Yields:
                Tuple of (input_X, target_y) for each microbatch
            """
            for batch_id in range(self.get_num_batches()):
                yield from self.batch_iterator(batch_id)

        def load_micro_batch_input(self, batch_id: int, mubatch_id: int) -> np.ndarray:
            """Load input data for a specific microbatch (legacy method)"""
            assert 0 <= batch_id < self.get_num_batches(), "Invalid batch_id"
            assert 0 <= mubatch_id < self.get_num_mubatches(), "Invalid mubatch_id"
            
            start_idx = batch_id * self.local_batch_size + mubatch_id * self.mubatch_size
            end_idx = start_idx + self.mubatch_size
            
            return self.input_X[start_idx:end_idx]
    
        def load_micro_batch_target(self, batch_id: int, mubatch_id: int) -> np.ndarray:
            """Load target data for a specific microbatch (legacy method)"""
            assert 0 <= batch_id < self.get_num_batches(), "Invalid batch_id"
            assert 0 <= mubatch_id < self.get_num_mubatches(), "Invalid mubatch_id"
            
            start_idx = batch_id * self.local_batch_size + mubatch_id * self.mubatch_size
            end_idx = start_idx + self.mubatch_size
            
            return self.target_y[start_idx:end_idx]


    




        

         






