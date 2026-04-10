from ..utils.objects.metrics import Metrics
import torch
import time
import numpy as np
from ..utils import log as logger


class Train(object):
    def __init__(self, step, epochs, verbose=False):
        self.epochs = epochs
        self.step = step
        self.history = History()
        self.verbose = verbose

    def __call__(self, train_loader_step, val_loader_step=None, early_stopping=None, current_epoch=1):
        # We only run 1 epoch per call because main.py handles the outer loop
        for _ in range(self.epochs):
            self.step.train()
            train_stats = train_loader_step(self.step)
            # Update history with the actual epoch number from main.py
            self.history(train_stats, current_epoch)

            if val_loader_step is not None:
                with torch.no_grad():
                    self.step.eval()
                    val_stats = val_loader_step(self.step)
                    self.history(val_stats, current_epoch)

            if self.verbose:
                print(self.history)

            if early_stopping is not None and val_loader_step is not None:
                valid_loss = val_stats.loss()
                # If early stopping triggers, return True to signal main.py to stop
                if early_stopping(valid_loss):
                    return True
        return False


def _collect_probs_labels(step, loader_step):
    all_probs, all_labels = [], []
    device = getattr(step, 'device', 'cpu')

    with torch.no_grad():
        step.model.eval()
        for batch in loader_step.loader:
            data_input = batch[0] if isinstance(batch, (tuple, list)) else batch
            if hasattr(data_input, 'x'):
                data_input = data_input.to(device)
                probs = torch.sigmoid(step.model(data_input)).view(-1)
                all_probs.append(probs.detach().cpu())
                all_labels.append(data_input.y.detach().cpu().float().view(-1))

    if not all_probs:
        return None, None
    return torch.cat(all_probs), torch.cat(all_labels)


def _find_best_threshold_from_val(val_probs, val_labels):
    best_t, best_acc = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.01):
        pred = (val_probs >= t).float()
        acc = (pred == val_labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def predict(step, test_loader_step, val_loader_step=None, threshold=0.5):
    print("Testing")

    chosen_threshold = threshold
    if val_loader_step is not None:
        vp, vy = _collect_probs_labels(step, val_loader_step)
        if vp is not None:
            chosen_threshold, val_acc = _find_best_threshold_from_val(vp, vy)
            print(f"  Threshold tuning: selected threshold={chosen_threshold:.2f}, val accuracy={val_acc:.4f}")

    with torch.no_grad():
        step.model.eval()
        all_outs, all_labels = [], []
        device = getattr(step, 'device', 'cpu')

        for batch in test_loader_step.loader:
            data_input = batch[0] if isinstance(batch, (tuple, list)) else batch
            if hasattr(data_input, 'x'):
                data_input = data_input.to(device)
                out = torch.sigmoid(step.model(data_input)).view(-1)
                all_outs.append(out.detach().cpu())
                all_labels.append(data_input.y.detach().cpu().float().view(-1))

        if len(all_outs) == 0:
            print("Error: No valid graph data found.")
            return 0.0

        flat_outs = torch.cat(all_outs)
        flat_labels = torch.cat(all_labels)

        print(f"  Prediction diagnostics:")
        print(f"    Mean: {flat_outs.mean():.4f} | Std: {flat_outs.std():.4f} | Min: {flat_outs.min():.4f} | Max: {flat_outs.max():.4f}")
        print(f"    Class distribution in test — Neg: {(flat_labels == 0).sum().item()} | Pos: {(flat_labels == 1).sum().item()}")

        pred_labels = (flat_outs >= chosen_threshold).float()
        metrics = Metrics(pred_labels, flat_labels)
        print(metrics)
        metrics.log()

    return metrics()["Accuracy"]

class History:
    def __init__(self):
        self.history = {}
        self.epoch = 0
        self.timer = time.time()

    def __call__(self, stats, epoch):
        self.epoch = epoch
        # We overwrite the list so we only show the LATEST chunk's performance in console
        self.history[epoch] = [stats]

    def __str__(self):
        # Prevents the "growing" string problem
        stats = ' - '.join([f"{res}" for res in self.current()])
        return f"{stats}"

    def current(self):
        return self.history.get(self.epoch, ["No Stats"])

    def log(self):
        msg = f"(Epoch: {self.epoch}) {' - '.join([f'({res})' for res in self.current()])}"
        logger.log_info("history", msg)