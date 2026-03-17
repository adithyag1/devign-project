from ..utils.objects.metrics import Metrics
import torch
import time
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

def predict(step, test_loader_step):
    print(f"Testing")
    with torch.no_grad():
        step.model.eval()
        all_outs = []
        all_labels = []
        device = getattr(step, 'device', 'cpu') 

        for batch in test_loader_step.loader:
            data_input = batch[0] if isinstance(batch, (tuple, list)) else batch
            if hasattr(data_input, 'x'):
                data_input = data_input.to(device)
                output = torch.sigmoid(step.model(data_input))
                all_outs.append(output.detach().cpu())
                all_labels.append(data_input.y.detach().cpu())
            
        if len(all_outs) == 0:
            print("Error: No valid graph data found.")
            return 0.0

        flat_outs = torch.cat(all_outs, dim=0)
        flat_labels = torch.cat(all_labels, dim=0)
        
        metrics = Metrics(flat_outs, flat_labels)
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