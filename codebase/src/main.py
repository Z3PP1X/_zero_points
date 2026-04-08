import mlflow
from models import TestGraphNetwork
from preprocessing import GraphPipeline
import torch
from torch import optim
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from dataset import DatasetDescriptor

# --- Config ---
# Wir setzen die Threads explizit auf 6, um deine Ressourcenplanung einzuhalten
NUM_CORES = 6
torch.set_num_threads(NUM_CORES)

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

DATASET_NAME = "run_20260408_160456/dataset_4"
EXPERIMENTS_DIR = "../../_datasets/run_20260408_160456/graphs"
SEED = 42001
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
SAVE_PATH = "../../_models/best_model.pth"

# Metriken initialisieren
f1_metric = MulticlassF1Score(num_classes=2).to(DEVICE)
precision_metric = MulticlassPrecision(num_classes=2).to(DEVICE)
recall_metric = MulticlassRecall(num_classes=2).to(DEVICE)

mlflow.set_tracking_uri("http://localhost:5000")


def create_experiment_name():
    return DATASET_NAME + "_" + str(SEED) + "_" + str(EPOCHS)


def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.batch, batch.global_features)
        loss = criterion(outputs, batch.y.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(
                batch.x, batch.edge_index, batch.batch, batch.global_features
            )
            labels = batch.y.squeeze()
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            f1_metric.update(probs, labels)
            precision_metric.update(probs, labels)
            recall_metric.update(probs, labels)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    f1_computed = f1_metric.compute().item()
    precision_computed = precision_metric.compute().item()
    recall_computed = recall_metric.compute().item()

    return avg_loss, accuracy, f1_computed, precision_computed, recall_computed


def main():
    pipeline = GraphPipeline(
        dataset_name=DATASET_NAME,
        experiments_dir=EXPERIMENTS_DIR,
        seed=SEED,
    )

    # Nutze num_workers=NUM_CORES für paralleles Laden der Graphen auf der CPU
    train_loader, test_loader, class_weights = pipeline.pipe(
        test_size=TEST_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=3,
    )

    model = TestGraphNetwork.from_pipeline(pipeline).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    mlflow.set_experiment(DATASET_NAME)

    with mlflow.start_run(run_name=create_experiment_name()):
        mlflow.log_params(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "test_size": TEST_SIZE,
                "device": DEVICE,
                "num_threads": NUM_CORES,
                "model": "TestGraphNetwork (GATv2)",
                "input_dim": pipeline.input_dim,
                "global_dim": pipeline.global_dim,
            }
        )

        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss, val_acc, f1_val, prec_val, rec_val = evaluate(
                model, test_loader, criterion
            )

            mlflow.log_metrics(
                {
                    "Loss/train": train_loss,
                    "Loss/val": val_loss,
                    "Accuracy/val": val_acc,
                    "F1": f1_val,
                    "Precision": prec_val,
                    "Recall": rec_val,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"F1: {f1_val:.4f} | "
                f"Precision: {prec_val:.4f} | "
                f"Recall: {rec_val:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVE_PATH)
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                print(f"  ↳ Saved best model (val_loss={val_loss:.4f})")

        mlflow.pytorch.log_model(model, "model")

    print("Training complete.")


if __name__ == "__main__":
    dataset_des = DatasetDescriptor(DATASET_NAME)
    dataset_des._load_dataset()
    print(dataset_des.pandas_dataframe.tail())
    main()
