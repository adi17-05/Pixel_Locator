
import numpy as np
import matplotlib.pyplot as plt

def show_worst_cases_reg(dataset, model, steps, title="Worst cases", k=6):
    """
    Visualizes the worst k predictions (highest Euclidean distance error)
    from the regression model.

    Args:
        dataset: A tf.data.Dataset object.
        model: The trained regression model.
        steps: Number of batches to iterate.
        title: Title for the plot.
        k: Number of worst cases to show.
    """
    # Assuming predict_on_reg_dataset is available or imported. 
    # If not, we would need to pass it or define it here.
    # For now, we assume it's in the same scope or imported.
    # We will define a placeholder or rely on the user to have it.
    # To be safe, let's assume it's available in the notebook context.
    
    y_true_all, y_pred_all = predict_on_reg_dataset(model, dataset, steps)
    dist = np.sqrt(np.sum((y_true_all - y_pred_all) ** 2, axis=1))

    # Get indices of k worst
    worst_idx = np.argsort(dist)[-k:]

    # Collect images
    # We iterate the dataset again to get the images.
    # This assumes deterministic ordering or shuffled with same seed/state if reset.
    images = []
    it = iter(dataset)
    for _ in range(steps):
        x_batch, _ = next(it)
        images.append(x_batch.numpy())
    all_images = np.vstack(images)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(worst_idx):
        plt.subplot(1, k, i + 1)
        # Squeeze to handle (H, W, 1) -> (H, W) for grayscale
        img = all_images[idx]
        plt.imshow(img.squeeze(), cmap="gray")

        gt = y_true_all[idx]
        pred = y_pred_all[idx]
        
        # Plot Ground Truth
        plt.scatter(gt[0], gt[1], c="g", s=20, label="GT")
        # Plot Prediction
        plt.scatter(pred[0], pred[1], c="r", s=20, label="Pred")

        plt.title(f"Err: {dist[idx]:.2f}px")
        if i == 0:
            plt.legend()
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
