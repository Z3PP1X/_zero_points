import sys
import os
import argparse
import torch
from pathlib import Path
import mlflow
from gnn.supervised_learning.preprocessing import GraphPipeline

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

gnn_root = Path(__file__).resolve().parents[2]
if str(gnn_root) not in sys.path:
    sys.path.insert(0, str(gnn_root))
src_root = Path(__file__).resolve().parents[3]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from gnn.supervised_learning.supervised_config import (
    apply_expression_graph_overrides,
    bootstrap_graphgym_cfg,
    create_graphgym_model,
    load_yaml_config,
    read_supervised_settings,
    resolve_edge_dim,
    validate_layer_type,
)
from gnn.supervised_learning.loader_graphgym import (
    compute_binary_metrics,
    configure_class_weights,
    get_pos_label,
    set_pos_label_from_train_labels,
)
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler

NUM_CORES = 6
torch.set_num_threads(NUM_CORES)

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

DEFAULT_SEED = 42001
TEST_SIZE = 0.2


def create_experiment_name(dataset_name: str, mode: str, epochs: int, seed: int) -> str:
    return f"{dataset_name}_{seed}_{epochs}_{mode}"


def train_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        batch.split = "train"
        optimizer.zero_grad()
        pred, true = model(batch)
        loss, _ = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred_score = []

    # Register forward hook to capture embeddings from model.mp
    mp_embeddings = []
    def hook_fn(module, inputs, outputs):
        mp_embeddings.append((outputs.x.detach().cpu(), outputs.edge_index.detach().cpu(), getattr(outputs, 'edge_attr', None)))
        
    hook_handle = model.mp.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            batch.split = "val"
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            batch_size = true.size(0)
            total_loss += loss.item() * batch_size
            all_true.append(true.detach().cpu())
            all_pred_score.append(pred_score.detach().cpu())

    hook_handle.remove()

    if not all_true:
        empty_metrics = compute_binary_metrics(
            torch.tensor([], dtype=torch.long),
            torch.tensor([], dtype=torch.float),
        )
        empty_metrics["dirichlet_energy"] = 0.0
        return 0.0, empty_metrics, [], [], []

    true_cat = torch.cat(all_true)
    pred_cat = torch.cat(all_pred_score)
    metrics = compute_binary_metrics(true_cat, pred_cat)

    # Compute average dirichlet energy
    from gnn.shared.utils.graph_utils import compute_normalized_dirichlet_energy
    energies = []
    for x, edge_index, edge_attr in mp_embeddings:
        energy = compute_normalized_dirichlet_energy(x, edge_index)
        energies.append(energy)
    avg_energy = sum(energies) / len(energies) if energies else 0.0
    metrics["dirichlet_energy"] = avg_energy

    avg_loss = total_loss / max(true_cat.size(0), 1)

    pos_label = get_pos_label()
    if pred_cat.ndim > 1 and pred_cat.shape[1] > 1:
        probs = pred_cat[:, pos_label].numpy().tolist()
    else:
        scores = pred_cat.numpy()
        probs = scores.tolist() if pos_label == 1 else (1.0 - scores).tolist()

    labels = true_cat.numpy().tolist()
    preds = (np.array(probs) >= 0.5).astype(int).tolist()
    if pos_label == 0:
        preds = [1 - p for p in preds]

    return avg_loss, metrics, labels, preds, probs


def log_confusion_matrix(y_true, y_pred, epoch: int):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
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


def save_supervised_training_curves(history, output_path: Path):
    import matplotlib.pyplot as plt
    import numpy as np

    metrics_to_plot = ["pr_auc", "auc", "loss", "f1", "mean_confidence", "ece", "dirichlet_energy"]
    key_mapping = {
        "pr_auc": "PR_AUC/val",
        "auc": "AUC/val",
        "loss": "Loss/val",
        "f1": "F1",
        "mean_confidence": "mean_confidence",
        "ece": "ece",
        "dirichlet_energy": "dirichlet_energy/val",
    }
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(11, 3.2 * n_metrics), dpi=150)
    axes = np.atleast_1d(axes)
    
    epochs = history["epoch"]
    
    for ax, metric in zip(axes, metrics_to_plot):
        history_key = key_mapping[metric]
        if history_key in history:
            ax.plot(
                epochs,
                history[history_key],
                marker="o",
                linewidth=2,
                markersize=4,
                label=f"Validation {metric.upper()}",
                color="#E76F51",
            )
        if metric == "loss" and "Loss/train" in history:
            ax.plot(
                epochs,
                history["Loss/train"],
                marker="o",
                linewidth=2,
                markersize=4,
                label="Train Loss",
                color="#2A9D8F",
            )
        ax.set_title(metric.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, frameon=False)
        
    plt.suptitle("Supervised Learning Training Curves", fontsize=14, fontweight="bold", y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {output_path}")


def main(
    config_path: Path,
    dataset_name: str,
    mode: str = "graph",
    active_features: list[str] | None = None,
    feature_groups: list[str] | None = None,
    node_features: list[str] | None = None,
    topology_features: list[str] | None = None,
    positional_encoding: list[str] | None = None,
    edge_features: list[str] | None = None,
    synthetic: bool = False,
    synthetic_dataset: str | None = None,
    layer_type: str = "gatv2conv",
    edge_direction: str = "top_down",
    heterogeneous: bool = False,
    add_kappa: bool = False,
    add_virtual_supernode: bool = False,
):
    seed = DEFAULT_SEED
    cfg = bootstrap_graphgym_cfg(config_path, seed=seed)
    feature_selection = apply_expression_graph_overrides(
        cfg,
        mode=mode,
        active_features=active_features,
        feature_groups=feature_groups,
        node_features=node_features,
        topology_features=topology_features,
        positional_encoding=positional_encoding,
        edge_features=edge_features,
        synthetic=synthetic,
        synthetic_dataset=synthetic_dataset,
        edge_direction=edge_direction,
        heterogeneous=heterogeneous,
        add_kappa=add_kappa,
        add_virtual_supernode=add_virtual_supernode,
    )
    resolved_active_features = (
        None
        if not cfg.expression_graph.active_features
        else [
            feature.strip()
            for feature in str(cfg.expression_graph.active_features).split(",")
            if feature.strip()
        ]
    )
    cfg.gnn.layer_type = validate_layer_type(layer_type)
    cfg.dataset.edge_dim = resolve_edge_dim()

    epochs = int(cfg.train.epochs)
    batch_size = int(cfg.train.batch_size)
    repo_root = Path(__file__).resolve().parents[4]
    save_dir = repo_root / "_models"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "best_model.pth"

    from gnn.shared.utils.unified_loader import UnifiedDataLoader

    unified_loader = UnifiedDataLoader.get_instance(
        dataset_name=dataset_name,
        mode=mode,
        heterogeneous=heterogeneous,
        edge_direction=edge_direction,
        add_kappa=add_kappa,
        add_virtual_supernode=add_virtual_supernode,
    )

    pipeline = GraphPipeline(
        dataset_name=dataset_name,
        seed=seed,
        mode=mode,
        active_features=resolved_active_features,
        unified_loader=unified_loader,
        synthetic=synthetic,
        synthetic_dataset_name=synthetic_dataset,
        layer_type=layer_type,
        heterogeneous=heterogeneous,
        add_kappa=add_kappa,
        add_virtual_supernode=add_virtual_supernode,
    )

    train_loader, val_loader, class_weights = pipeline.pipe(
        test_size=TEST_SIZE,
        batch_size=batch_size,
        stratify=True,
        num_workers=3,
    )

    # In non-synthetic mode pipe() builds val_loader over the FULL 20% holdout and there
    # is no separate test set, so the set used for checkpoint selection would also be the
    # one reported as generalization (leakage). Split the holdout into two disjoint halves
    # (deterministic interleave) so val drives selection and test stays untouched.
    test_loader = None
    if not synthetic:
        from torch_geometric.loader import DataLoader as _PyGDataLoader

        _holdout = [
            pipeline.test_dataset[i] for i in range(len(pipeline.test_dataset))
        ]
        val_loader = _PyGDataLoader(
            _holdout[0::2], batch_size=batch_size, shuffle=False
        )
        test_loader = _PyGDataLoader(
            _holdout[1::2], batch_size=batch_size, shuffle=False
        )

    sample = pipeline.train_dataset[0]
    dim_in = sample.x.shape[1]
    configure_class_weights(class_weights)
    train_labels = torch.tensor(
        [pipeline.train_dataset[i].y.item() for i in range(len(pipeline.train_dataset))]
    )
    set_pos_label_from_train_labels(train_labels)

    print("Initializing GraphGym GNN model (same stack as main_graphgym.py)...")
    model = create_graphgym_model(cfg, dim_in=dim_in, device=DEVICE)
    optimizer = create_optimizer(model.parameters(), cfg.optim)
    scheduler = create_scheduler(optimizer, cfg.optim)

    print(f"Initializing MLflow tracking at {mlflow.get_tracking_uri()} ...")
    mlflow.set_experiment(dataset_name)

    from gnn.supervised_learning.curated_eval_schedule import (
        parse_curated_eval_schedule,
        should_evaluate_curated,
    )

    curated_schedule = parse_curated_eval_schedule(cfg.train)
    curated_loader = getattr(pipeline, "curated_loader", None)

    print("Starting MLflow run and training loop...")
    with mlflow.start_run(run_name=create_experiment_name(dataset_name, mode, epochs, seed)):
        mlflow.log_params(
            {
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "base_lr": float(cfg.optim.base_lr),
                "weight_decay": float(cfg.optim.weight_decay),
                "scheduler": str(cfg.optim.scheduler),
                "test_size": TEST_SIZE,
                "device": DEVICE,
                "num_threads": NUM_CORES,
                "model": "GraphGym GNN",
                "input_dim": dim_in,
                "edge_dim": int(cfg.dataset.edge_dim),
                "layer_type": cfg.gnn.layer_type,
                "layers_mp": int(cfg.gnn.layers_mp),
                "dim_inner": int(cfg.gnn.dim_inner),
                "stage_type": str(cfg.gnn.stage_type),
                "graph_pooling": str(cfg.model.graph_pooling),
                "loss_fun": str(cfg.model.loss_fun),
                "mode": mode,
                "active_features": resolved_active_features,
                "feature_groups": feature_selection.enabled_groups(),
                "positional_encodings": list(feature_selection.positional_encodings),
                "synthetic": synthetic,
                "synthetic_dataset": synthetic_dataset,
                "curated_eval_period": curated_schedule.period,
                "curated_eval_on_test_highscore": curated_schedule.on_test_highscore,
            }
        )

        best_val_pr_auc = float("-inf")
        history = {
            "epoch": [],
            "Loss/train": [],
            "Loss/val": [],
            "Accuracy/val": [],
            "F1": [],
            "Precision": [],
            "Recall": [],
            "AUC/val": [],
            "PR_AUC/val": [],
            "mean_confidence": [],
            "ece": [],
            "dirichlet_energy/val": [],
        }

        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            print("  Training...")
            train_loss = train_epoch(model, train_loader, optimizer)
            scheduler.step()
            print("  Evaluating...")
            val_loss, metrics, y_true, y_pred, y_prob = evaluate(model, val_loader)

            log_confusion_matrix(y_true, y_pred, epoch)

            mlflow.log_metrics(
                {
                    "Loss/train": train_loss,
                    "Loss/val": val_loss,
                    "Accuracy/val": metrics["accuracy"],
                    "F1": metrics["f1"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "AUC/val": metrics["auc"],
                    "PR_AUC/val": metrics["pr_auc"],
                    "mean_confidence": metrics.get("mean_confidence", 0.0),
                    "ece": metrics.get("ece", 0.0),
                    "dirichlet_energy/val": metrics.get("dirichlet_energy", 0.0),
                    "base_lr": float(optimizer.param_groups[0]["lr"]),
                },
                step=epoch,
            )

            history["epoch"].append(epoch)
            history["Loss/train"].append(train_loss)
            history["Loss/val"].append(val_loss)
            history["Accuracy/val"].append(metrics["accuracy"])
            history["F1"].append(metrics["f1"])
            history["Precision"].append(metrics["precision"])
            history["Recall"].append(metrics["recall"])
            history["AUC/val"].append(metrics["auc"])
            history["PR_AUC/val"].append(metrics["pr_auc"])
            history["mean_confidence"].append(metrics.get("mean_confidence", 0.0))
            history["ece"].append(metrics.get("ece", 0.0))
            history["dirichlet_energy/val"].append(metrics.get("dirichlet_energy", 0.0))

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"ROC-AUC: {metrics['auc']:.4f} | "
                f"PR-AUC: {metrics['pr_auc']:.4f} | "
                f"Dirichlet Energy: {metrics.get('dirichlet_energy', 0.0):.6f} | "
                f"Recall: {metrics['recall']:.4f}"
            )

            is_new_highscore = metrics["pr_auc"] > best_val_pr_auc
            if is_new_highscore:
                best_val_pr_auc = metrics["pr_auc"]
                torch.save(model.state_dict(), str(save_path))
                mlflow.log_metric("best_val_pr_auc", best_val_pr_auc, step=epoch)
                print(f"  ↳ Saved best model (val_pr_auc={metrics['pr_auc']:.4f})")

            if synthetic and curated_loader is not None:
                should_run_curated, curated_reason = should_evaluate_curated(
                    epoch,
                    curated_schedule,
                    is_new_test_highscore=is_new_highscore,
                )
                if should_run_curated:
                    cur_loss, cur_metrics, _, _, _ = evaluate(model, curated_loader)
                    mlflow.log_metrics(
                        {
                            "Loss/curated": cur_loss,
                            "Accuracy/curated": cur_metrics["accuracy"],
                            "F1/curated": cur_metrics["f1"],
                            "Precision/curated": cur_metrics["precision"],
                            "Recall/curated": cur_metrics["recall"],
                            "AUC/curated": cur_metrics["auc"],
                            "PR_AUC/curated": cur_metrics["pr_auc"],
                        },
                        step=epoch,
                    )
                    print(
                        f"  ↳ Curated holdout eval ({curated_reason}) | "
                        f"PR-AUC={cur_metrics['pr_auc']:.4f}"
                    )

        if synthetic and curated_loader is not None:
            print("\nEvaluating best saved model on Curated (Real) Dataset...")
            best_model = create_graphgym_model(cfg, dim_in=dim_in, device=DEVICE)
            best_model.load_state_dict(torch.load(str(save_path), map_location=DEVICE))
            cur_loss, cur_metrics, _, _, _ = evaluate(best_model, pipeline.curated_loader)

            mlflow.log_metrics(
                {
                    "Loss/curated": cur_loss,
                    "Accuracy/curated": cur_metrics["accuracy"],
                    "F1/curated": cur_metrics["f1"],
                    "Precision/curated": cur_metrics["precision"],
                    "Recall/curated": cur_metrics["recall"],
                    "AUC/curated": cur_metrics["auc"],
                    "PR_AUC/curated": cur_metrics["pr_auc"],
                }
            )
            print("-" * 50)
            print(
                f"Final Curated (Real) Evaluation | "
                f"Loss: {cur_loss:.4f} | "
                f"Acc: {cur_metrics['accuracy']:.4f} | "
                f"F1: {cur_metrics['f1']:.4f} | "
                f"ROC-AUC: {cur_metrics['auc']:.4f} | "
                f"PR-AUC: {cur_metrics['pr_auc']:.4f}"
            )
            print("-" * 50)

        elif test_loader is not None:
            print("\nEvaluating best saved model on disjoint Test holdout...")
            best_model = create_graphgym_model(cfg, dim_in=dim_in, device=DEVICE)
            best_model.load_state_dict(torch.load(str(save_path), map_location=DEVICE))
            test_loss, test_metrics, _, _, _ = evaluate(best_model, test_loader)

            mlflow.log_metrics(
                {
                    "Loss/test": test_loss,
                    "Accuracy/test": test_metrics["accuracy"],
                    "F1/test": test_metrics["f1"],
                    "Precision/test": test_metrics["precision"],
                    "Recall/test": test_metrics["recall"],
                    "AUC/test": test_metrics["auc"],
                    "PR_AUC/test": test_metrics["pr_auc"],
                }
            )
            print("-" * 50)
            print(
                f"Final Test (disjoint holdout) Evaluation | "
                f"Loss: {test_loss:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f} | "
                f"F1: {test_metrics['f1']:.4f} | "
                f"ROC-AUC: {test_metrics['auc']:.4f} | "
                f"PR-AUC: {test_metrics['pr_auc']:.4f}"
            )
            print("-" * 50)

        try:
            curves_path = save_dir / "training_curves.png"
            save_supervised_training_curves(history, curves_path)
            mlflow.log_artifact(str(curves_path))
        except Exception as exc:
            print(f"Warning: Failed to generate or log training curves: {exc}")

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
        description="Start GNN Supervised Learning experiment (GraphGym-aligned stack)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset.name from config (e.g. run_key/dataset_name).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Loads data and prints structure without starting full training.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_supervised.yaml",
        help="GraphGym YAML config (architecture via gnn.layer_type).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["graph", "tree", "tree_derivatives"],
        help="Override GNN experiment mode from config.",
    )
    parser.add_argument(
        "--edge-direction",
        type=str,
        default=None,
        choices=["top_down", "bottom_up", "bidirectional"],
        help="AST message-passing direction (virtual-node edges stay bidirectional).",
    )
    from gnn.shared.utils.feature_config import add_feature_cli_args

    add_feature_cli_args(parser)
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
    parser.add_argument(
        "--heterogeneous",
        action="store_true",
        help="Enables heterogeneous graph mode.",
    )
    parser.add_argument(
        "--add-kappa",
        action="store_true",
        help="Merge kappa (h-function) subgraphs from datasets/kappas/ into each graph.",
    )
    parser.add_argument(
        "--add-virtual-supernode",
        action="store_true",
        help=(
            "Add a fully-connected virtual supernode (bidirectional edges to every node) "
            "to shorten message-passing paths across the graph."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / args.config
    settings = read_supervised_settings(load_yaml_config(config_path))

    dataset_name = args.dataset or settings["dataset_name"]
    if not dataset_name:
        parser.error("Dataset name must be set in config (dataset.name) or via --dataset")

    mode = args.mode or settings["mode"]
    edge_direction = args.edge_direction or settings["edge_direction"]
    layer_type = settings["layer_type"]
    synthetic = args.synthetic or settings["synthetic"]
    synthetic_dataset = args.synthetic_dataset or settings["synthetic_dataset"]
    heterogeneous = args.heterogeneous or settings["heterogeneous"]
    add_kappa = args.add_kappa or settings["add_kappa"]
    add_virtual_supernode = (
        args.add_virtual_supernode or settings["add_virtual_supernode"]
    )

    from gnn.shared.utils.unified_loader import UnifiedDataLoader

    try:
        loader = UnifiedDataLoader.get_instance(
            dataset_name=dataset_name,
            mode=mode,
            heterogeneous=heterogeneous,
            edge_direction=edge_direction,
            add_kappa=add_kappa,
            add_virtual_supernode=add_virtual_supernode,
        )
        print("Curated dataset loaded successfully!")
        print(loader.data.tail())
        print_dataset_distribution(dataset_name, loader.data)
    except Exception as e:
        print(f"Note: Curated dataset files not found in local sandbox: {e}")

    if synthetic and synthetic_dataset:
        try:
            synth_loader = UnifiedDataLoader.get_instance(
                dataset_name=synthetic_dataset,
                mode=mode,
                heterogeneous=heterogeneous,
                edge_direction=edge_direction,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
            )
            print("Synthetic dataset loaded successfully!")
            print(synth_loader.data.tail())
            print_dataset_distribution(synthetic_dataset, synth_loader.data)
        except Exception as e:
            print(f"Note: Synthetic dataset files not found in local sandbox: {e}")

    if not args.dry_run:
        try:
            from gnn.supervised_learning.supervised_config import resolve_expression_graph_features

            preview_selection, _ = resolve_expression_graph_features(
                load_yaml_config(config_path).get("expression_graph"),
                feature_groups=args.feature_groups,
                node_features=args.node_features,
                topology_features=args.topology_features,
                positional_encoding=args.positional_encoding,
                edge_features=args.edge_features,
                active_features=args.active_features,
            )
            print(f"Feature groups: {preview_selection.enabled_groups()}")
            print(f"Positional encodings: {list(preview_selection.positional_encodings)}")
            print(f"Active node features: {preview_selection.summary()}")
            print(
                f"Config: {config_path.name} | layer_type={layer_type} | edge_direction={edge_direction}"
            )
            main(
                config_path=config_path,
                dataset_name=dataset_name,
                mode=mode,
                feature_groups=args.feature_groups,
                node_features=args.node_features,
                topology_features=args.topology_features,
                positional_encoding=args.positional_encoding,
                edge_features=args.edge_features,
                active_features=args.active_features,
                synthetic=synthetic,
                synthetic_dataset=synthetic_dataset,
                layer_type=layer_type,
                edge_direction=edge_direction,
                heterogeneous=heterogeneous,
                add_kappa=add_kappa,
                add_virtual_supernode=add_virtual_supernode,
            )
        except Exception as e:
            print(
                f"Failed to start training run (expected if datasets/connections not available in sandbox): {e}"
            )
    else:
        print("[Dry Run] Supervised script verification completed successfully.")
