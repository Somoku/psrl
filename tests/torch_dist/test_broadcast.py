import os

import torch
import torch.cuda
import torch.distributed as dist


# Initialization function
def setup():
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl")

    # Get the local rank and set the corresponding GPU device
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)


# Main function to perform the broadcast
def main():
    # Initialize the distributed setup
    setup()

    # Create a group of specific ranks (for example, ranks 1 and 2)
    group = dist.new_group(ranks=[0, 2, 3])

    # Model version to broadcast
    broadcast_model_version = torch.tensor([22], dtype=torch.long).cuda()

    # Broadcast to the selected ranks
    if dist.get_rank() == 0:
        # Rank 0 will broadcast to ranks 1 and 2
        dist.broadcast(tensor=broadcast_model_version, src=0, group=group)
        print(f"Rank {dist.get_rank()} broadcasted version: {broadcast_model_version.item()}")
    else:
        # Non-representative ranks will receive the broadcasted model version
        broadcast_model_version = torch.zeros(1, dtype=torch.long).cuda()
        dist.broadcast(tensor=broadcast_model_version, src=0, group=group)
        print(f"Rank {dist.get_rank()} received version: {broadcast_model_version.item()}")


if __name__ == "__main__":
    main()
