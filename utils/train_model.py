import time
import torch

from utils.misc import set_seed
from utils.regret import regret
from utils.create_loss_func import create_loss_func


def train_model(reg, optmodel, method, num_epochs=100, lr=0.001, train_dataloader=None, val_dataloader=None,
                hyperparameters=None, seed=0, adj_vert_computation_time=0,
                log_every_n_epochs=1, mm=None):
    """
    Train a model using stochastic gradient descent.
    
    Args:
        reg: The regression model to train
        optmodel: The optimization model
        method: The training method/loss function to use
        num_epochs: Number of epochs to train for
        lr: Learning rate
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        hyperparameters: Additional hyperparameters for the loss function
        seed: Random seed
        adj_vert_computation_time: Time taken to compute adjacent vertices
        log_every_n_epochs: How often to log training progress
    
    Returns:
        tuple: (loss_log_regret, val_loss_log_regret, train_time)
    """
    set_seed(seed)
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    reg.train()
    initial_regret = regret(reg, optmodel, train_dataloader)
    initial_val_regret = regret(reg, optmodel, val_dataloader)
    loss_log_regret = [initial_regret]
    val_loss_log_regret = [initial_val_regret]
    print(f"Start training method {method}")
    print(f"Before training, regret: {initial_regret}")
    loss_func = create_loss_func(method, optmodel, hyperparameters, train_dataloader.dataset)
    train_time = 0

    if "adjacent_vertices" in method:
        train_time += adj_vert_computation_time

    # Track best model state
    best_val_regret = float('inf')
    best_epoch = 0
    best_model_state = None
    best_train_time = 0

    # Early stopping variables
    patience = 3  # Number of checks to wait for improvement
    min_improvement = 0.01  # 1% improvement threshold
    patience_counter = 0
    best_val_regret_for_stopping = None  # Initialize as None

    max_training_time = 600

    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            # Unpack data
            x, c, w, z, w_rel, z_rel, bctr, adj_verts = data

            # Make prediction
            cp = reg(x)
            tic = time.time()

            # Handle different loss functions
            if method == "mse":
                loss = loss_func(cp, c)

            elif method == "cosine":
                loss = (torch.ones(size=(cp.shape[0],)) - loss_func(cp, c)).mean()

            elif method == "spo_plus":
                loss = loss_func(cp, c, w, z).mean()

            elif method == "pfyl":
                loss = loss_func(cp, w).mean()

            elif method == "nce":
                loss = loss_func(cp, w).mean()

            elif method in ["inner_cave", "exact_cave"]:
                loss = loss_func(cp, bctr)

            elif method == "negative_identity":
                pred_sols = loss_func(cp)
                loss = torch.mean((w - pred_sols) ** 2)

            elif method == "lava":
                loss = loss_func(cp, adj_verts, w_rel, mm)

            else:
                raise Exception("Unknown loss type")

            toc = time.time()
            train_time += toc - tic

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % log_every_n_epochs == 0) or epoch == num_epochs - 1:
            train_regret = regret(reg, optmodel, train_dataloader)
            val_regret = regret(reg, optmodel, val_dataloader)
            loss_log_regret.append(train_regret)
            val_loss_log_regret.append(val_regret)
            print(f"Epoch: {epoch}, regret: {train_regret}")
            print(f"Epoch: {epoch}, val regret: {val_regret}\n")

            # Track best model state
            if val_regret < best_val_regret:
                best_val_regret = val_regret
                best_epoch = epoch
                best_model_state = {k: v.clone() for k, v in reg.state_dict().items()}
                best_train_time = train_time

            # Early stopping check
            if best_val_regret_for_stopping is None:
                best_val_regret_for_stopping = val_regret
                patience_counter = 0
            else:
                improvement = (best_val_regret_for_stopping - val_regret) / best_val_regret_for_stopping
                if improvement > min_improvement:
                    patience_counter = 0
                    best_val_regret_for_stopping = val_regret
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch} - No {min_improvement * 100}% improvement in validation regret for {patience} consecutive checks")
                break

        if train_time > max_training_time:
            # Do a final validation check before stopping
            train_regret = regret(reg, optmodel, train_dataloader)
            val_regret = regret(reg, optmodel, val_dataloader)
            loss_log_regret.append(train_regret)
            val_loss_log_regret.append(val_regret)
            print(f"Final check - Epoch: {epoch}, regret: {train_regret}")
            print(f"Final check - Epoch: {epoch}, val regret: {val_regret}\n")

            # Update best model if this final check is the best
            if val_regret < best_val_regret:
                best_val_regret = val_regret
                best_epoch = epoch
                best_model_state = {k: v.clone() for k, v in reg.state_dict().items()}
                best_train_time = max_training_time

            print(f"Stopping training because training time exceeded {max_training_time} seconds")
            break

        if patience_counter >= patience:
            break

    # Restore best model state
    if best_model_state is not None:
        reg.load_state_dict(best_model_state)
        train_time = best_train_time
        print(f"Restored model state from epoch {best_epoch} with validation regret: {best_val_regret:.4f}")

    return loss_log_regret, val_loss_log_regret, train_time
