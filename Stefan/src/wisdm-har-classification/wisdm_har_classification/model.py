import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from wisdm_har_classification.logger import logger


class Net(torch.nn.Module):
    def __init__(self, input_dim=43, num_classes=6):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, 46)
        x = x.reshape(x.size(0), -1)  # flatten (46)
        return self.model(x)


def set_weights(model, weights):
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w).reshape(p.shape)


def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.parameters()]


def train(model, trainloader, epochs, device, num_classes=6):
    model.train()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    last_loss = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        last_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{epochs} completed. Train Loss: {last_loss:.4f}")

    # ---- training metrics ----
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, pred = torch.max(out, 1)

            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)

    return {
        "acc": accuracy,
        "loss": last_loss,
    }


def test(model, valloader, device, num_classes=6):
    model.eval()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    loss_total = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in valloader:
            X, y = X.to(device), y.to(device)

            out = model(X)
            loss_total += criterion(out, y).item()

            _, pred = torch.max(out, 1)

            correct += (pred == y).sum().item()
            total += y.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = loss_total / len(valloader)
    accuracy = correct / total

    # per-class metrics
    precision = precision_score(
        all_targets,
        all_preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    recall = recall_score(
        all_targets,
        all_preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )
    f1 = f1_score(
        all_targets,
        all_preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

    metrics = {
        "acc": accuracy,
        "loss": avg_loss,
    }

    for i in range(num_classes):
        metrics[f"p_{i}"] = precision[i]
        metrics[f"r_{i}"] = recall[i]
        metrics[f"f1_{i}"] = f1[i]

        for j in range(num_classes):
             metrics[f"cm_{i}_{j}"] = int(cm[i, j])

    return metrics
