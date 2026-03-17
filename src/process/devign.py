import torch.optim as optim
import torch.nn.functional as F
import torch
from ..utils import log
from .step import Step

class Devign(Step):
    def __init__(self,
                 path: str,
                 device: str,
                 model: object, 
                 learning_rate: float,
                 weight_decay: float,
                 loss_lambda: float,
                 weight_0: float,
                 weight_1: float):
        self.path = path
        self.lr = learning_rate
        self.wd = weight_decay
        self.ll = loss_lambda
        self.w0 = weight_0
        self.w1 = weight_1
        
        log.log_info('devign', f"LR: {self.lr}; WD: {self.wd}; LL: {self.ll}; W0: {self.w0}; W1: {self.w1}")
        
        # FIX: Define _model by moving the passed model to the device
        _model = model.to(device)
        
        # Define weights and loss logic
        def weighted_loss(o, t):
            t = t.view_as(o)
            weights = torch.where(t == 0, self.w0, self.w1).to(o.device)
            return F.binary_cross_entropy_with_logits(o, t, weight=weights) + F.l1_loss(o, t) * self.ll

        self.optimizer = optim.Adam(_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        super().__init__(model=_model,
                         loss_function=weighted_loss,
                         optimizer=self.optimizer
                         )

        self.count_parameters()

    def load(self):
        try:
            self.model.load_state_dict(torch.load(self.path, map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'), weights_only=True))
            print(f"Weights loaded from {self.path}")
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch.")

    def save(self):
        import os
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(self.model.state_dict(), self.path)

    def count_parameters(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"The model has {count:,} trainable parameters")