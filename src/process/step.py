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
    def __init__(self, model, loss_function, optimizer, w0=1.0, w1=1.0, accumulation_steps=1):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.w0 = w0
        self.w1 = w1
        self.accumulation_steps = max(1, int(accumulation_steps))
        self.clip_grad_norm = 5.0
        self._accum_count = 0
        self.optimizer.zero_grad()
        self.max_grad_norm = 0

    def __call__(self, i, batch_data, y):
        logits = self.model(batch_data).view(-1)
        target = y.float()

        # label smoothing
        smooth = 0.05
        target_smoothed = target * (1.0 - smooth) + 0.5 * smooth

        probs = torch.sigmoid(logits)

        # plain BCE-with-logits (no class-weight scaling)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_smoothed)

        acc = binary_accuracy(probs, target)

        if self.model.training:
            # Scale loss by accumulation steps before backward
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()
            self._accum_count += 1

            if self._accum_count % self.accumulation_steps == 0:
                if self.clip_grad_norm > 0:
                    # ✅ clip_grad_norm_ returns the total norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    ).item()
                    self.max_grad_norm = max(self.max_grad_norm, grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

        # IMPORTANT: Return probs (0-1), not logits (-inf to +inf)
        return stats.Stat(probs.tolist(), loss.item(), acc.item(), y.tolist())
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
