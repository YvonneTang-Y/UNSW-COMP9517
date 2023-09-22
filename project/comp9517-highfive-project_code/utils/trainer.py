import torch
from tqdm import tqdm

def unbatch(batch, device):
    """
    Unbatches a batch of data from the Dataloader.
    Inputs
        batch: tuple
            Tuple containing a batch from the Dataloader.
        device: str
            Indicates which device (CPU/GPU) to use.
    Returns
        X: list
            List of images.
        y: list
            List of dictionaries.
    """
    X, y = batch
    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y

class Trainer:
  def __init__(
    self,
    epochs,
    train_loader,
    optimizer,
    device,
    valid_loader = None,
    criterion = None,
    save_dir = None
  ):
    self.epochs = epochs
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.device = device
    self.save_dir = save_dir

    # self.logger = get_logger(str(Path(self.save_dir).joinpath("log.txt")))
    self.best_loss = float("inf")

  def fit(self, model):
    for epoch in range(self.epochs):
      model.train()

      with tqdm(self.train_loader, dynamic_ncols=True) as pbar:
        pbar.set_description(f"[Epoch {epoch + 1}/{self.epochs}]")

        for batch in pbar:
          X, y = unbatch(batch, device = self.device)
          self.optimizer.zero_grad()
          losses = model(X, y)
          loss = sum(loss for loss in losses.values()) / len(y)
          loss.backward()

          self.optimizer.step()

          pbar.set_postfix(loss=loss)

      print(f"[Train] epoch: {epoch} loss: {loss}")
      # self.logger.info(f"(train) epoch: {epoch} loss: {losses.avg}")
      self.evaluate(model, epoch)

  @torch.no_grad()
  def predict(self, model, data_loader):
    """
    Gets the predictions for a batch of data.
    Inputs
      model: torch model
      data_loader: torch Dataloader
      device: str
        Indicates which device (CPU/GPU) to use.
    Returns
      images: list
        List of tensors of the images.
      predictions: list
        List of dicts containing the predictions for the 
        bounding boxes, labels and confidence scores.
    """
    model.eval()

    images = []
    predictions = []

    for i, batch in enumerate(data_loader):
      X, _ = unbatch(batch, device = self.device)
      p = model(X)
      images = images + X
      predictions = predictions + p
    
    return images, predictions

  @torch.no_grad()
  def evaluate(self, model, epoch):
    model.train()

    for batch in tqdm(self.valid_loader):
      X, y = unbatch(batch, device = self.device)
      self.optimizer.zero_grad()
      losses = model(X, y)
      loss = sum(loss for loss in losses.values()) / len(y)

    print(f"[Evaluation] epoch: {epoch} loss: {loss}")

    # if loss <= self.best_loss:
    #   self.best_acc = loss
    torch.save(model.state_dict(), self.save_dir)
