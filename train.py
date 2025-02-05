import argparse
import torch
import torch.optim as optim
from src.model import DyAb
from src.main_utils import save_model, get_device
from src.train_utils import get_data_loader, step
import wandb
import uuid
import os


def main():
    parser = argparse.ArgumentParser(description="DyAb reimp XDD")
    parser.add_argument("--img_dim", type=int, help="Image size for resnet")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Set device (default: cpu)")
    args = parser.parse_args()

    wandb.init(project="DyAb reboot")
    
    device = get_device('cuda')
    model = DyAb(img_dim=args.img_dim, device=args.device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    embedding_dictionary  = torch.load("embeddings/ablang.pt")

    train_loader = get_data_loader(embedding_dictionary, "data/train.csv", args.batch_size, shuffle=True)
    test_loader = get_data_loader(embedding_dictionary, "data/test.csv", args.batch_size, shuffle=False)


    model_checkpoint_directory = "model_checkpoints"
    os.makedirs(model_checkpoint_directory, exist_ok=True)

    # Create a unique subfolder for each run to avoid name collisions
    unique_id = uuid.uuid4().hex[:8]
    model_checkpoint_run_directory = os.path.join(model_checkpoint_directory, unique_id)
    os.makedirs(model_checkpoint_run_directory, exist_ok=True)
    best_model_save_path = os.path.join(model_checkpoint_run_directory, "best_model.pth")


    max_metric = 0
    for epoch in range(1, args.epochs+1):
        train_loss, train_mae, train_r2, train_spearman = step(model, train_loader, device, criterion, optimizer, train=True)
        test_loss, test_mae, test_r2, test_spearman = step(model, train_loader, device, criterion, optimizer=None, train=False)

        wandb.log({
            "train/loss": train_loss,
            "train/mae": train_mae,
            "train/r2": train_r2,
            "train/spearman": train_spearman,
            "test/loss": test_loss,
            "test/mae": test_mae,
            "test/r2": test_r2,
            "test/spearman": test_spearman
        })

        watch_metric = test_spearman
        if watch_metric > max_metric:
            max_metric = watch_metric
            save_model(model, best_model_save_path)

    wandb.finish()

if __name__ == "__main__":
    main()
