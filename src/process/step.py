import torch
from ..utils.objects import stats
import configs

def binary_accuracy(probs, labels):
    # Round probabilities to nearest integer (0 or 1)
    preds = torch.round(probs)
    # Compare with ground truth
    correct = (preds == labels).sum().float()
    acc = correct / len(labels)
    return acc


class Step:
    def __init__(self, model, loss_function, optimizer, w0=1.0, w1=1.0):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.w0 = w0
        self.w1 = w1
        # Gradient accumulation: update weights every N batches
        self.accumulation_steps = 1
        # Gradient clipping max norm (0 disables clipping)
        self.clip_grad_norm = 1.0
        # Internal counter for accumulation
        self._accum_count = 0
        # Start with clean gradients
        self.optimizer.zero_grad()

    def __call__(self, i, batch_data, y):
        # 1. The model now returns raw LOGITS (no sigmoid)
        logits = self.model(batch_data).view(-1)
        target = y.float()
        
        # 2. Convert logits to probabilities for stats/accuracy
        # We need this because stats.Stat and binary_accuracy expect 0.0 to 1.0
        probs = torch.sigmoid(logits)
        
        # 3. Weighted Loss Logic using the Logit-stable function
        loss_weights = torch.where(target == 0, self.w0, self.w1).to(target.device)
        
        # Use binary_cross_entropy_with_logits for numerical stability
        raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
        loss = (raw_loss * loss_weights).mean()
        
        # 4. Accuracy and Optimizer
        acc = binary_accuracy(probs, target)

        if self.model.training:
            # Scale loss by accumulation steps before backward
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()
            self._accum_count += 1

            if self._accum_count % self.accumulation_steps == 0:
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # IMPORTANT: Return probs (0-1), not logits (-inf to +inf)
        return stats.Stat(probs.tolist(), loss.item(), acc.item(), y.tolist())
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
