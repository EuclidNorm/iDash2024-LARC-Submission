import argparse
import os
import torch
from flamby.utils import evaluate_model_on_tests
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds,
    FedTcgaBrca,
)
from flamby.strategies.fed_opt import FedAdam as strat

def create_train_dataloaders(num_clients, batch_size):
    return [
        torch.utils.data.DataLoader(
            FedTcgaBrca(center=i),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        for i in range(num_clients)
    ]

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataloaders = create_train_dataloaders(NUM_CLIENTS, BATCH_SIZE)
    lossfunc = BaselineLoss()
    model = Baseline().to(device)
    
    strategy_args = {
        "training_dataloaders": train_dataloaders,
        "model": model,
        "loss": lossfunc,
        "optimizer_class": torch.optim.SGD,
        "learning_rate": LR,
        "num_updates": 100,
        "nrounds": get_nb_max_rounds(100),
    }
    
    strategy = strat(**strategy_args)
    trained_model = strategy.run()[0]

    train_dataset = FedTcgaBrca(train=True, pooled=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers
    )

    train_cindex = evaluate_model_on_tests(trained_model, [train_dataloader], metric)
    print("C-index on training set after training:", train_cindex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU to run the training on (if available)")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers for the dataloader")
    args = parser.parse_args()

    main(args)
