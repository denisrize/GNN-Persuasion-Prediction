from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import matplotlib.pyplot as plt


def mask_author_nodes(predictions, labels, node_types):
    # Create a mask to exclude author nodes
    mask = (node_types != 0).to(torch.bool)

    # Apply the mask to predictions and labels
    predictions = predictions[mask]
    labels = labels[mask]
    return predictions, labels

def train_and_evaluate_model(
    model, dataloaders, epochs=10, lr=0.01, class_weights=None, weight_decay=0.0, device="cpu", patience=20
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)

    # Use class weights in the loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_losses, val_losses = [], []

    best_val_loss = float("inf")  # Initialize the best validation loss
    best_model_state = None       # Variable to store the best model state
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in dataloaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(batch)

            predictions_masked, labels_masked = mask_author_nodes(out, batch.y, batch.node_type)
            loss = loss_fn(predictions_masked, labels_masked)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in dataloaders["val"]:
                batch = batch.to(device)
                out = model(batch)
                predictions_masked, labels_masked = mask_author_nodes(out, batch.y, batch.node_type)
                loss = loss_fn(predictions_masked, labels_masked)
                val_loss += loss.item()

            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model if this is the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model's state
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return train_losses, val_losses, best_model_state

def evaluate_model_on_test(model, dataloader, results_dir="results", device="cpu"):
    model.eval()
    model = model.to(device)  # Ensure model is on the correct device
    with torch.no_grad():
        all_preds, all_labels, all_probs = [], [], []

        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch)

            # Extract predictions, probabilities, and labels
            probs = torch.softmax(out, dim=1)[:, 1]  # Probabilities for the positive class
            preds = out.argmax(dim=1)
            labels = batch.y

            # Mask author nodes using your mask function
            probs_masked, labels_masked = mask_author_nodes(probs, labels, batch.node_type)
            preds_masked, _ = mask_author_nodes(preds, labels, batch.node_type)

            # Convert to numpy for metric calculations
            all_probs.extend(probs_masked.cpu().numpy())
            all_preds.extend(preds_masked.cpu().numpy())
            all_labels.extend(labels_masked.cpu().numpy())

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs)  # Calculate ROC-AUC score
        print(f"Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format="d")

        # Save Confusion Matrix
        plt.title("Confusion Matrix")
        plt.savefig(f"{results_dir}/confusion_matrix.png")
        plt.close()

        # Classification Report
        report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
        print(report)

        # Save Classification Report
        with open(f"{results_dir}/classification_report.txt", "w") as f:
            f.write(report)

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plot_roc(fpr, tpr, roc_auc, results_dir=results_dir)

def calculate_class_weights(dataloader, device="cpu"):
    """
    Calculate class weights for the training dataset.
    """
    all_labels = []
    for batch in dataloader:
        all_labels.extend(batch.y.cpu().numpy())

    # Convert class labels to NumPy array
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),  # Convert classes to NumPy array
        y=np.array(all_labels)    # Convert labels to NumPy array
    )
    return torch.tensor(class_weights, dtype=torch.float).to(device)

def plot_loss(train_losses, val_losses, results_dir="results"):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f"{results_dir}/loss_curve.png")
    plt.close()
    
def plot_roc(fpr, tpr, roc_auc, results_dir="results"):
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{results_dir}/roc_curve.png")
    plt.close()