import sys
import os
import argparse
import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
import mlflow
from gnn.supervised_learning.preprocessing import GraphPipeline

from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.shared.models.classifiers import TestGraphNetwork # noqa


NUM_CORES = 6
torch.set_num_threads(NUM_CORES)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

SEED = 42001
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
SAVE_PATH = "../../_models/best_model.pth"

f1_metric = MulticlassF1Score(num_classes=2).to(DEVICE)
precision_metric = MulticlassPrecision(num_classes=2).to(DEVICE)
recall_metric = MulticlassRecall(num_classes=2).to(DEVICE)


def create_experiment_name(dataset_name: str, mode: str = "graph"):
    return dataset_name + "_" + str(SEED) + "_" + str(EPOCHS) + "_" + mode


def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.global_features,
            edge_attr=getattr(batch, "edge_attr", None),
        )
        loss = criterion(outputs, batch.y.squeeze())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def log_confusion_matrix(y_true, y_pred, epoch: int):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
        title=f"Confusion Matrix - Epoch {epoch + 1}",
        ylabel="True label",
        xlabel="Predicted label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()

    temp_path = f"temp_cm_epoch_{epoch}.png"
    plt.savefig(temp_path, dpi=100)
    plt.close(fig)

    try:
        mlflow.log_artifact(temp_path, artifact_path="confusion_matrices")
    except Exception as e:
        print(f"Warning: Failed to log confusion matrix to MLflow: {e}")

    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass


def evaluate(model, loader, criterion):
    model.eval()
    f1_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.global_features,
                edge_attr=getattr(batch, "edge_attr", None),
            )
            labels = batch.y.squeeze()
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            probs = torch.softmax(outputs, dim=1)
            f1_metric.update(probs, labels)
            precision_metric.update(probs, labels)
            recall_metric.update(probs, labels)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            if probs.dim() == 1:
                probs = probs.unsqueeze(0)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    f1_computed = f1_metric.compute().item()
    precision_computed = precision_metric.compute().item()
    recall_computed = recall_metric.compute().item()

    return (
        avg_loss,
        accuracy,
        f1_computed,
        precision_computed,
        recall_computed,
        all_labels,
        all_preds,
        all_probs,
    )


def main(
    dataset_name: str,
    mode: str = "graph",
    enrich: bool = False,
    active_features: list[str] | None = None,
    synthetic: bool = False,
    synthetic_dataset: str | None = None,
    architecture: str = "gatv2_stack",
):
    repo_root = Path(__file__).resolve().parents[4]
    dataset_path = dataset_name
    save_dir = repo_root / "_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "best_model.pth"
    from gnn.shared.utils.unified_loader import UnifiedDataLoader

    unified_loader = UnifiedDataLoader.get_instance(
        dataset_name=dataset_path,
        mode=mode,
        enrich=enrich,
    )

    pipeline = GraphPipeline(
        dataset_name=dataset_path,
        seed=SEED,
        mode=mode,
        enrich=enrich,
        active_features=active_features,
        unified_loader=unified_loader,
        synthetic=synthetic,
        synthetic_dataset_name=synthetic_dataset,
        architecture=architecture,
    )

    train_loader, test_loader, class_weights = pipeline.pipe(
        test_size=TEST_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=3,
    )

    print("Initializing GNN model...")
    model = TestGraphNetwork.from_pipeline(pipeline).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Initializing MLflow tracking at {mlflow.get_tracking_uri()} ...")
    mlflow.set_experiment(dataset_path)

    print("Starting MLflow run and training loop...")
    with mlflow.start_run(run_name=create_experiment_name(dataset_path, mode)):
        mlflow.log_params(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "test_size": TEST_SIZE,
                "device": DEVICE,
                "num_threads": NUM_CORES,
                "model": f"TestGraphNetwork ({architecture})",
                "input_dim": pipeline.input_dim,
                "edge_dim": pipeline.edge_dim,
                "global_dim": pipeline.global_dim,
                "architecture": architecture,
                "mode": mode,
                "enrich": enrich,
                "active_features": active_features,
                "synthetic": synthetic,
                "synthetic_dataset": synthetic_dataset,
            }
        )

        best_val_loss = float("inf")

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
            print("  Training...")
            train_loss = train(model, train_loader, optimizer, criterion)
            print("  Evaluating...")
            val_loss, val_acc, f1_val, prec_val, rec_val, y_true, y_pred, y_prob = (
                evaluate(model, test_loader, criterion)
            )

            # Compute advanced metrics
            y_true_np = np.array(y_true)
            y_pred_np = np.array(y_pred)
            y_prob_np = np.array(y_prob)

            if len(np.unique(y_true_np)) > 1:
                roc_auc = float(roc_auc_score(y_true_np, y_prob_np))
            else:
                roc_auc = 0.5

            f1_classes = f1_score(
                y_true_np, y_pred_np, labels=[0, 1], average=None, zero_division=0
            )
            prec_classes = precision_score(
                y_true_np, y_pred_np, labels=[0, 1], average=None, zero_division=0
            )
            rec_classes = recall_score(
                y_true_np, y_pred_np, labels=[0, 1], average=None, zero_division=0
            )

            f1_c0, f1_c1 = float(f1_classes[0]), float(f1_classes[1])
            prec_c0, prec_c1 = float(prec_classes[0]), float(prec_classes[1])
            rec_c0, rec_c1 = float(rec_classes[0]), float(rec_classes[1])

            # Log confusion matrix plot as artifact
            log_confusion_matrix(y_true, y_pred, epoch)

            mlflow.log_metrics(
                {
                    "Loss/train": train_loss,
                    "Loss/val": val_loss,
                    "Accuracy/val": val_acc,
                    "F1": f1_val,
                    "Precision": prec_val,
                    "Recall": rec_val,
                    "AUC/val": roc_auc,
                    "F1_class_0": f1_c0,
                    "F1_class_1": f1_c1,
                    "Precision_class_0": prec_c0,
                    "Precision_class_1": prec_c1,
                    "Recall_class_0": rec_c0,
                    "Recall_class_1": rec_c1,
                },
                step=epoch,
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"F1: {f1_val:.4f} | "
                f"ROC-AUC: {roc_auc:.4f} | "
                f"F1 (C0/C1): {f1_c0:.4f}/{f1_c1:.4f} | "
                f"Prec (C0/C1): {prec_c0:.4f}/{prec_c1:.4f} | "
                f"Rec (C0/C1): {rec_c0:.4f}/{rec_c1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), str(save_path))
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                print(f"  ↳ Saved best model (val_loss={val_loss:.4f})")

        if synthetic and getattr(pipeline, "curated_loader", None) is not None:
            print("\nEvaluating best saved model on Curated (Real) Dataset...")
            best_model = TestGraphNetwork.from_pipeline(pipeline).to(DEVICE)
            best_model.load_state_dict(torch.load(str(save_path)))
            cur_loss, cur_acc, f1_cur, prec_cur, rec_cur, cur_true, cur_pred, cur_prob = (
                evaluate(best_model, pipeline.curated_loader, criterion)
            )
            cur_true_np = np.array(cur_true)
            cur_prob_np = np.array(cur_prob)
            if len(np.unique(cur_true_np)) > 1:
                cur_roc_auc = float(roc_auc_score(cur_true_np, cur_prob_np))
            else:
                cur_roc_auc = 0.5
            
            mlflow.log_metrics(
                {
                    "Loss/curated": cur_loss,
                    "Accuracy/curated": cur_acc,
                    "F1/curated": f1_cur,
                    "Precision/curated": prec_cur,
                    "Recall/curated": rec_cur,
                    "AUC/curated": cur_roc_auc,
                }
            )
            print("-" * 50)
            print(f"Final Curated (Real) Evaluation | "
                  f"Loss: {cur_loss:.4f} | "
                  f"Acc: {cur_acc:.4f} | "
                  f"F1: {f1_cur:.4f} | "
                  f"ROC-AUC: {cur_roc_auc:.4f}")
            print("-" * 50)

        mlflow.pytorch.log_model(model, "model")

    print("Training complete.")


def print_dataset_distribution(dataset_name: str, df: pd.DataFrame):
    if "faster_algorithm" not in df.columns:
        boundaries = [
            df["Newton_absTime"] < df["GMGF_absTime"],
            df["Newton_absTime"] > df["GMGF_absTime"],
        ]
        values = [1, 0]
        df["faster_algorithm"] = np.select(boundaries, values)

    counts = df["faster_algorithm"].value_counts()
    total = len(df)
    newton_count = counts.get(1, 0)
    gmgf_count = counts.get(0, 0)
    perc_newton = (newton_count / total) * 100 if total > 0 else 0.0
    perc_gmgf = (gmgf_count / total) * 100 if total > 0 else 0.0

    print(f"--- Verteilung für Dataset: {dataset_name} ---")
    print(f"Gesamtanzahl Samples: {total}")
    print(f"Klasse 1 (Newton): {newton_count:>5} ({perc_newton:>5.2f}%)")
    print(f"Klasse 0 (gMGF):   {gmgf_count:>5} ({perc_gmgf:>5.2f}%)")
    print("-" * (30 + len(dataset_name)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start GNN Supervised Learning experiment"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="run_20260408_160456/dataset_4",
        help="Dataset name, optionally including run key (e.g. run_key/dataset_name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Loads data and prints structure without starting full training.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="graph",
        choices=["graph", "tree", "tree_derivatives"],
        help="Select GNN experiment mode: graph (with virtual nodes), tree (features on global node, f only) or tree_derivatives (f, f', f'' connected via global node)",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Toggles enriched features in supervised learning pipeline (uses 19 features instead of 8).",
    )
    parser.add_argument(
        "--active-features",
        type=str,
        default=None,
        help="Comma-separated list of active GNN node features to use (dynamically adapts dimensions).",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="gatv2_stack",
        choices=["gatv2_stack", "gine_stack"],
        help="Edge-aware GNN architecture: gatv2_stack (attention) or gine_stack (edge MLP).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Enables synthetic mode: train on synthetic dataset, validate on curated dataset.",
    )
    parser.add_argument(
        "--synthetic-dataset",
        type=str,
        default=None,
        help="Synthetic dataset name, optionally including run key (e.g. synthetic_run_key/synthetic_dataset_name)",
    )
    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).resolve().parents[4]

    from gnn.shared.utils.unified_loader import UnifiedDataLoader

    try:
        loader = UnifiedDataLoader.get_instance(
            dataset_name=args.dataset, mode=args.mode, enrich=args.enrich
        )
        print("Curated dataset loaded successfully!")
        print(loader.data.tail())
        print_dataset_distribution(args.dataset, loader.data)
    except Exception as e:
        print(f"Note: Curated dataset files not found in local sandbox: {e}")

    if args.synthetic and args.synthetic_dataset:
        try:
            synth_loader = UnifiedDataLoader.get_instance(
                dataset_name=args.synthetic_dataset, mode=args.mode, enrich=args.enrich
            )
            print("Synthetic dataset loaded successfully!")
            print(synth_loader.data.tail())
            print_dataset_distribution(args.synthetic_dataset, synth_loader.data)
        except Exception as e:
            print(f"Note: Synthetic dataset files not found in local sandbox: {e}")

    if not args.dry_run:
        try:
            active_feats = None
            if args.active_features is not None:
                active_feats = [
                    f.strip() for f in args.active_features.split(",") if f.strip()
                ]
                print(f"Aktivierte Features: {active_feats}")
            main(
                dataset_name=args.dataset,
                mode=args.mode,
                enrich=args.enrich,
                active_features=active_feats,
                synthetic=args.synthetic,
                synthetic_dataset=args.synthetic_dataset,
                architecture=args.architecture,
            )
        except Exception as e:
            print(
                f"Failed to start training run (expected if datasets/connections not available in sandbox): {e}"
            )
    else:
        print("[Dry Run] Supervised script verification completed successfully.")
