import argparse
import time
import threading
from pathlib import Path

import numpy as np
from mpi4py import MPI

from sanghathan.dataset import Dataset
from sanghathan.layers import MLP
from sanghathan.optimizer import SGD
from sanghathan.pipe import (
    GPipeSchedule,
    InferenceSchedule,
    NaiveParallelSchedule,
    PipeDreamSchedule,
    Worker,
)
from sanghathan.utils import assert_sync, get_model_hash
# Import visualization server components
try:
    from sanghathan.server import (
        start_monitoring_server,
        update_training_state,
        register_process,
        update_metrics,
        mark_training_complete
    )
    VISUALIZATION_ENABLED = True
except ImportError:
    VISUALIZATION_ENABLED = False
    print("Visualization server not available. Run without monitoring.")


def compute_accuracy(model, worker, dataset):
    """
    Compute accuracy using a safer approach
    """
    model.eval()

    try:
        correct = 0
        total = 0
        
        # Make sure processes are synchronized 
        if worker.pp_comm is not None:
            worker.pp_comm.Barrier()
        
        compute_start = time.time()
        
        for batch_id in range(dataset.get_num_batches()):
            schedule = InferenceSchedule(
                num_micro_batches=1,
                num_stages=worker.pipeline_depth,
                stage_id=worker.stage_id,
            )
            
            worker.execute(schedule, batch_id)
            
            # Synchronize after each batch
            if worker.pp_comm is not None:
                worker.pp_comm.Barrier()

            if worker.stage_id == worker.pipeline_depth - 1:
                pred = np.argmax(worker.output_buffers[0], axis=-1)
                target = np.argmax(dataset.load_micro_batch_target(batch_id, 0), axis=-1)
                correct += np.sum(pred == target)
                total += pred.shape[0]
        
        compute_time = time.time() - compute_start
        model.train()
        
        # One final barrier to ensure all processes are done
        if worker.pp_comm is not None:
            worker.pp_comm.Barrier()
            
        if worker.stage_id == worker.pipeline_depth - 1:
            accuracy = correct / total
            return accuracy, compute_time
        return None, compute_time
    except Exception as e:
        print(f"Error computing accuracy: {str(e)}")
        if worker.pp_comm is not None:
            worker.pp_comm.Abort()
        return None, 0


SCHEDULE_NAME_TO_CLS = {
    "naive": NaiveParallelSchedule,
    "gpipe": GPipeSchedule,
    "pipedream": PipeDreamSchedule,
}

EPOCHS = 20
# We use a big batch size, to make training more amenable to parallelization
GLOBAL_BATCH_SIZE = 128
N_MUBATCHES = 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Degree of data parallelism (=number of full model replicas)",
    )
    parser.add_argument("--pp", type=int, default=1, help="Number of pipeline stages")
    parser.add_argument(
        "--schedule", type=str, choices=["pipedream", "gpipe", "naive"], default="naive"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization server"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for visualization server"
    )
    args = parser.parse_args()
    DP_tile_factor = args.dp
    PP_tile_factor = args.pp

    assert DP_tile_factor >= 1 and PP_tile_factor >= 1
    assert DP_tile_factor * PP_tile_factor == MPI.COMM_WORLD.size, (
        f"Number of started workers is {MPI.COMM_WORLD.size}, "
        f"but should be {DP_tile_factor * PP_tile_factor} (DP * PP)"
    )
    assert (
        GLOBAL_BATCH_SIZE % DP_tile_factor == 0
    ), "Batch size must be properly divisible by DP"

    # create MPI communicators for data parallel AllReduce & pipeline parallel send & recv
    # if the `color=` parameter is the same, then those two workers end up in the same communicator
    dp_comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() % PP_tile_factor)
    # to run it truly distributed (like on a RaspberryPi cluster) you'd use comm.Split_type
    # instead of this color splitting, eg TYPE_SOCKET for PP
    pp_comm = MPI.COMM_WORLD.Split(color=MPI.COMM_WORLD.Get_rank() // PP_tile_factor)
    # sanity check
    assert dp_comm.Get_size() == DP_tile_factor and pp_comm.Get_size() == PP_tile_factor

    # Set up the local model.
    # Layer_sizes is the total model size, which we split into PP-many stages
    layer_sizes = [784, 128, 127, 126, 125, 124, 123, 10]
    model = MLP(
        layer_sizes,
        stage_idx=pp_comm.rank,
        n_stages=PP_tile_factor,
        batch_size=GLOBAL_BATCH_SIZE,
    )
    model.train()

    optimizer = SGD(model.parameters(), lr=0.006)

    # Initialize visualization server (only on rank 0)
    if VISUALIZATION_ENABLED and args.visualize and MPI.COMM_WORLD.Get_rank() == 0:
        server_thread = threading.Thread(
            target=start_monitoring_server,
            args=(args.port,),
            daemon=True
        )
        server_thread.start()
        print(f"Visualization server started at http://localhost:{args.port}")
        
    # Register this process with the visualization system
    if VISUALIZATION_ENABLED and args.visualize:
        # Get a string representation of model layers
        model_layers_str = str(model).replace('\n', ' ')
        
        # Register this process
        register_process(
            rank=MPI.COMM_WORLD.Get_rank(),
            world_size=MPI.COMM_WORLD.Get_size(),
            stage_id=pp_comm.rank,
            dp_rank=dp_comm.rank,
            layers=model_layers_str
        )
        
        # Set initial training state
        if MPI.COMM_WORLD.Get_rank() == 0:
            update_training_state(epoch=0, iteration=0, status='initializing')

    # Each DP-worker gets a slice of the global batch-size
    # TODO not every worker needs the dataset
    save_dir = Path("data/mnist_784/")
    local_batch_size = GLOBAL_BATCH_SIZE // DP_tile_factor
    dataset = Dataset(
        save_dir,
        global_batch_size=GLOBAL_BATCH_SIZE,
        mubatch_size=local_batch_size // N_MUBATCHES,
        validation=False,
    )
    dataset.load(dp_comm.Get_rank(), dp_comm.Get_size())
    worker = Worker(dp_comm, pp_comm, model, dataset, optimizer)

    val_dataset = Dataset(
        save_dir,
        global_batch_size=GLOBAL_BATCH_SIZE,
        mubatch_size=GLOBAL_BATCH_SIZE,
        validation=True,
    )
    val_dataset.load(DP_rank=0, DP_size=1)
    val_worker = Worker(None, pp_comm, model, val_dataset, None)

    # Update status to running
    if VISUALIZATION_ENABLED and args.visualize and MPI.COMM_WORLD.Get_rank() == 0:
        update_training_state(epoch=0, iteration=0, status='running')

    # Track metrics
    total_compute_time = 0
    total_comm_time = 0
    current_loss = 0

    start_time = time.time()
    for iteration in range(EPOCHS):
        epoch_start_time = time.time()
        
        # Compute and report accuracy
        accuracy_result, accuracy_compute_time = compute_accuracy(model, val_worker, val_dataset)
        
        if accuracy_result is not None:
            accuracy = accuracy_result * 100  # Convert to percentage
            print(
                f"Epoch: {iteration}, Time Spent: {time.time() - start_time:.2f}s, Accuracy: {accuracy:.2f}%",
            )
            
            # Report metrics to visualization server
            if VISUALIZATION_ENABLED and args.visualize:
                update_metrics(
                    rank=MPI.COMM_WORLD.Get_rank(),
                    metrics_dict={
                        'accuracy': accuracy,
                        'loss': current_loss,
                        'throughput': GLOBAL_BATCH_SIZE / (time.time() - epoch_start_time + 1e-6),
                        'compute_time': total_compute_time,
                        'comm_time': total_comm_time,
                        'utilization': total_compute_time / (total_compute_time + total_comm_time + 1e-6),
                    }
                )

        # Update training state with current epoch
        if VISUALIZATION_ENABLED and args.visualize and MPI.COMM_WORLD.Get_rank() == 0:
            update_training_state(epoch=iteration, iteration=0)

        # Train on batches
        batch_losses = []
        total_compute_time = 0
        total_comm_time = 0
        
        for batch_id in range(0, dataset.get_num_batches()):
            # Update iteration in visualization
            if VISUALIZATION_ENABLED and args.visualize and MPI.COMM_WORLD.Get_rank() == 0:
                update_training_state(epoch=iteration, iteration=batch_id)
                
            # Create schedule
            schedule = SCHEDULE_NAME_TO_CLS[args.schedule](
                num_micro_batches=N_MUBATCHES,
                num_stages=PP_tile_factor,
                stage_id=pp_comm.rank,
            )
            
            # Time the computation and communication
            compute_start = time.time()
            
            # Execute the schedule
            loss = worker.execute(schedule, batch_id)
            
            compute_end = time.time()
            compute_time = compute_end - compute_start
            
            # Measure communication time (AllReduce for DP)
            comm_start = time.time()
            if worker.dp_comm and worker.dp_comm.Get_size() > 1:
                worker.dp_comm.Barrier()
            comm_time = time.time() - comm_start
            
            # Accumulate times
            total_compute_time += compute_time
            total_comm_time += comm_time
            
            # Track loss
            if loss is not None:
                batch_losses.append(loss)
                current_loss = loss
                
            # Report per-batch metrics
            if VISUALIZATION_ENABLED and args.visualize and batch_id % 5 == 0:
                update_metrics(
                    rank=MPI.COMM_WORLD.Get_rank(),
                    metrics_dict={
                        'loss': loss if loss is not None else 0,
                        'compute_time': compute_time,
                        'comm_time': comm_time,
                        'utilization': compute_time / (compute_time + comm_time + 1e-6),
                    }
                )

        # Calculate average loss for the epoch
        if batch_losses:
            avg_loss = sum(batch_losses) / len(batch_losses)
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"Epoch {iteration} average loss: {avg_loss:.4f}")

    # Final accuracy
    accuracy_result, _ = compute_accuracy(model, val_worker, val_dataset)
    if accuracy_result is not None:
        accuracy = accuracy_result * 100  # Convert to percentage
        print(
            f"Epoch: {EPOCHS}, Time Spent: {time.time() - start_time:.2f}s, Accuracy: {accuracy:.2f}%",
        )
        
        # Final metrics update
        if VISUALIZATION_ENABLED and args.visualize:
            update_metrics(
                rank=MPI.COMM_WORLD.Get_rank(),
                metrics_dict={
                    'accuracy': accuracy,
                    'loss': current_loss,
                    'throughput': GLOBAL_BATCH_SIZE / ((time.time() - start_time) / EPOCHS),
                    'compute_time': total_compute_time,
                    'comm_time': total_comm_time,
                    'utilization': total_compute_time / (total_compute_time + total_comm_time + 1e-6),
                }
            )

    # Sanity check: Make sure data parallel replicas have the same model weights
    assert_sync(dp_comm, get_model_hash(model))
    
    # Mark training as complete
    if VISUALIZATION_ENABLED and args.visualize and MPI.COMM_WORLD.Get_rank() == 0:
        mark_training_complete(success=True)
