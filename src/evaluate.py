import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from .data import make_loaders, class_names
from .train import evaluate
from .utils import collect_predictions
from .model import SimpleCNN


def show_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}",
                     ha="center", va="center")

    plt.tight_layout()
    plt.show()


def evaluate_on_test(model, batch_size=256, data_dir="./data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = model.to(device)
    model.eval()

    _, _, test_loader = make_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    print("\nEvaluating test dataset...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size_actual = labels.size(0)

            total_loss += loss.item() * batch_size_actual
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size_actual

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Progress: {total_samples}/{len(test_loader.dataset)}"
                )

    test_loss = total_loss / total_samples
    test_acc = total_correct / total_samples

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=range(len(class_names))
    )

    class_correct = cm.diagonal()
    class_total = cm.sum(axis=1)

    print("\nClass-wise Accuracy:")
    for i, name in enumerate(class_names):
        accuracy = class_correct[i] / class_total[i]
        print(f"{name}: {accuracy * 100:.2f}%")

    normalized_cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=range(len(class_names)),
        normalize="true"
    )

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        labels=range(len(class_names)),
        target_names=class_names
    ))

    show_confusion_matrix(normalized_cm)

    return test_loss, test_acc, cm

if __name__ == "__main__":
    model = SimpleCNN(dropout=0.2)

    checkpoint_path = "models/cifar10_cnn_best.pth"

    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )

    evaluate_on_test(model, batch_size=256)