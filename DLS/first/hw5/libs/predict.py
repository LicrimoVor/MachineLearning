from torch import nn, utils, no_grad, cat


def predict(model: nn.Module, test_loader: utils.data.DataLoader):
    """Ответ модели."""
    with no_grad():
        logits = []

        for inputs in test_loader:
            inputs = inputs
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    probs = nn.functional.softmax(cat(logits), dim=-1).numpy()
    return probs
