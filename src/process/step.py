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

devign_config = configs.Devign()

class Step:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.w0 = devign_config.weight_0
        self.w1 = devign_config.weight_1

    def __call__(self, i, batch_data, y):
        # 1. The model now returns raw LOGITS (no sigmoid)
        logits = self.model(batch_data).view(-1)
        target = y.float()
        
        # 2. Convert logits to probabilities for stats/accuracy
        # We need this because stats.Stat and binary_accuracy expect 0.0 to 1.0
        probs = torch.sigmoid(logits)
        
        # 3. Weighted Loss Logic using the Logit-stable function
        # Weights: 2.0 for Clean (0), 1.0 for Vulnerable (1)
        loss_weights = torch.where(target == 0, self.w0, self.w1).to(target.device)
        
        # Use binary_cross_entropy_with_logits for numerical stability
        raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')
        loss = (raw_loss * loss_weights).mean()
        
        # 4. Accuracy and Optimizer
        acc = binary_accuracy(probs, target)

        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # IMPORTANT: Return probs (0-1), not logits (-inf to +inf)
        return stats.Stat(probs.tolist(), loss.item(), acc.item(), y.tolist())
    
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()