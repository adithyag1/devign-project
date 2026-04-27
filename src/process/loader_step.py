from ..utils.objects import stats
from tqdm import tqdm

class LoaderStep:
    def __init__(self, name, data_loader, device):
        self.name = name
        self.loader = data_loader
        self.size = len(data_loader)
        self.device = device

    def __call__(self, step):
        self.stats = stats.Stats(self.name)

        progress_bar = tqdm(self.loader, desc=f"{self.name} Step", leave=False)

        for i, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            stat: stats.Stat = step(i, batch, batch.y)
            self.stats(stat)

            progress_bar.set_postfix(loss=f"{stat.loss:.4f}")

        return self.stats
