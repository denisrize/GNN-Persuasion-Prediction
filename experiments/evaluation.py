from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import matplotlib.pyplot as plt


def mask_nodes(predictions, labels, mask_node):
    # Create a mask to exclude author nodes
    mask = (mask_node != 1).to(torch.bool)

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

            predictions_masked, labels_masked = mask_nodes(out, batch.y, batch.mask_node)

            # Check if the batch is empty after masking
            if predictions_masked.numel() == 0 or labels_masked.numel() == 0:
                continue  # Skip this batch

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
                predictions_masked, labels_masked = mask_nodes(out, batch.y, batch.mask_node)
                if predictions_masked.numel() == 0 or labels_masked.numel() == 0:
                    continue  # Skip this batch
                loss = loss_fn(predictions_masked, labels_masked)
                val_loss += loss.item()

            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model if this is the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the model's state
            epochs_without_improvement = 0  # Reset the counter
            best_val_epoch = epoch
        else:
            epochs_without_improvement += 1

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    return train_losses, val_losses, best_model_state, best_val_loss, best_val_epoch

def evaluate_model_on_test(
    model, dataloader, results_dir="results", filter_by_distance=None, device="cpu"
):
    """
    Evaluate the model on the test dataset with optional masking for distance filtering.
    
    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader for the test dataset.
        results_dir: Directory to save results like confusion matrix and ROC curve.
        filter_by_distance: Distance threshold for filtering nodes (optional).
        device: The device (CPU/GPU) to use for evaluation.
    """
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
            distances = batch.node_op_distance

            # Step 1: Mask irrelevant nodes
            mask = (batch.mask_node != 1).to(torch.bool)
            probs_masked = probs[mask]
            labels_masked = labels[mask]
            preds_masked = preds[mask]
            distances_masked = distances[mask]

            # Step 2: Further mask nodes by distance if required
            if filter_by_distance is not None:
                mask = (distances_masked >= filter_by_distance).to(torch.bool)
                probs_masked = probs_masked[mask]
                labels_masked = labels_masked[mask]
                preds_masked = preds_masked[mask]

            # Convert to numpy for metric calculations
            all_probs.extend(probs_masked.cpu().numpy())
            all_preds.extend(preds_masked.cpu().numpy())
            all_labels.extend(labels_masked.cpu().numpy())

        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Test Accuracy: {acc:.4f}, Test F1: {f1:.4f}")

        # Plot ROC Curve
        if len(set(all_labels)) < 2:
            # Only one class present; manually compute metrics
            unique_class = list(set(all_labels))[0]  # The single class present
            precision = precision_score(all_labels, all_preds, pos_label=unique_class, zero_division=0)
            recall = recall_score(all_labels, all_preds, pos_label=unique_class, zero_division=0)
            f1 = f1_score(all_labels, all_preds, pos_label=unique_class, zero_division=0)
            
            report = (
                f"Only one class present: {unique_class}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1-Score: {f1:.4f}\n"
                f"Accuracy: {acc:.4f}\n"
                f"Support: {len(all_labels)}"
            )
        else:
            roc_auc = roc_auc_score(all_labels, all_probs)  # Calculate ROC-AUC score
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plot_roc(fpr, tpr, roc_auc, filter_by_distance, results_dir=results_dir)
            # Classification Report
            report = classification_report(all_labels, all_preds, target_names=["Negative", "Positive"])
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap="Blues", values_format="d")

            # Save Confusion Matrix
            plt.title("Confusion Matrix")
            plt.savefig(f"{results_dir}/confusion_matrix_{filter_by_distance}_distance.png")
            plt.close()

        # Save Classification Report
        with open(f"{results_dir}/classification_report_{filter_by_distance}_distance.txt", "w") as f:
            f.write(report)

def calculate_class_weights(dataloader, balanced=True, device="cuda"):
    """
    Calculate class weights for the training dataset.
    If `balanced` is False, it assigns equal weights to both classes.
    For `balanced`, only considers labels where `node_mask == 0`.
    """
    if not balanced:
        return torch.tensor([1.0, 20.0], dtype=torch.float).to(device)  # Equal weights

    all_labels = []
    for batch in dataloader:
        # Include only labels where node_mask == 0
        mask = batch.mask_node == 0  # Select nodes with mask = 0
        filtered_labels = batch.y[mask].cpu().numpy()  # Filtered labels
        all_labels.extend(filtered_labels)

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
    
def plot_roc(fpr, tpr, roc_auc, filter_by_distance=None, results_dir="results"):
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"{results_dir}/roc_curve_{filter_by_distance}_distance.png")
    plt.close()