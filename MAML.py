import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc
from multiModalTrainingV2 import preprocess, MultimodalDataset, CrossAttention, MLP, MultiModalModel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure checkpoint directory exists
os.makedirs("MAML_Chckpt/checkpoints", exist_ok=True)

def reset_cuda_memory():
    """Reset CUDA memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    print("CUDA memory reset completed.")

def prepare_datasets(train_data, val_data, test_data):
    train_texts, train_images, train_labels = train_data
    val_texts, val_images, val_labels = val_data
    test_texts, test_images, test_labels = test_data

    train_dataset = MultimodalDataset(train_texts, train_images, train_labels)
    val_dataset = MultimodalDataset(val_texts, val_images, val_labels)
    test_dataset = MultimodalDataset(test_texts, test_images, test_labels)

    return train_dataset, val_dataset, test_dataset

def inner_loop_maml(model, train_dataset, val_dataset, test_dataset, category, num_epochs=10, lr=1e-3):
    reset_cuda_memory()  # Reset CUDA memory before starting inner loop
    print(f"\nStarting inner loop for category {category}")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.CrossEntropyLoss()
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Reduced batch size
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    best_test_acc = 0

    epoch_pbar = tqdm(range(num_epochs), desc=f"Category {category} Epochs")
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        train_pbar = tqdm(train_dataloader, desc=f"Training", leave=False)
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Clear cache after each batch
            torch.cuda.empty_cache()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({'loss': total_loss / (train_pbar.n + 1), 'accuracy': 100. * correct / total})

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_pbar = tqdm(val_dataloader, desc="Validating", leave=False)
        with torch.no_grad():  # Use torch.no_grad() for evaluation
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_pbar.set_postfix({'accuracy': 100. * val_correct / val_total})

        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.2f}%")

        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        test_pbar = tqdm(test_dataloader, desc="Testing", leave=False)
        with torch.no_grad():  # Use torch.no_grad() for evaluation
            for batch in test_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                test_pbar.set_postfix({'accuracy': 100. * test_correct / test_total})

        test_accuracy = 100 * test_correct / test_total
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {test_accuracy:.2f}%")

        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            print(f"New best test accuracy: {best_test_acc:.2f}%")

        # Gradual checkpointing
        if (epoch + 1) % 1 == 0:
            checkpoint_path = f"MAML_Chckpt/checkpoints/model{category}Epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if test_accuracy >= 95.0:
            print(f"Early stopping for category: {category}")
            break

        epoch_pbar.set_postfix({'best_test_acc': best_test_acc})

    print(f"Inner loop for category {category} completed. Best test accuracy: {best_test_acc:.2f}%")
    return model, best_test_acc

def outer_loop_maml(model, train_dataset, val_dataset, test_dataset, num_meta_epochs=5):
    reset_cuda_memory()  # Reset CUDA memory before starting outer loop
    meta_optimizer = optim.Adam(model.parameters(), lr=1e-4)

    meta_pbar = tqdm(range(num_meta_epochs), desc="Meta-training")
    for meta_epoch in meta_pbar:
        print(f"\nStarting meta-epoch {meta_epoch+1}/{num_meta_epochs}")
        meta_loss = 0
        category_pbar = tqdm(range(3), desc="Categories", leave=False)
        for category in category_pbar:
            print(f"\nProcessing category {category}")
            clone_model = type(model)().to(device)
            clone_model.load_state_dict(model.state_dict())
            updated_model, best_test_acc = inner_loop_maml(clone_model, train_dataset, val_dataset, test_dataset, category)

            # Compute meta-loss on a batch from the test set
            test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
            criterion = nn.CrossEntropyLoss()
            batch = next(iter(test_dataloader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            with torch.no_grad():  # Use torch.no_grad() for evaluation
                outputs = updated_model(input_ids, attention_mask, images)
            loss = criterion(outputs, labels)
            meta_loss += loss
            print(f"Meta-loss for category {category}: {loss.item():.4f}")

        # Update original model
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        print(f"Meta-epoch {meta_epoch+1}/{num_meta_epochs}, Total Meta-loss: {meta_loss.item():.4f}")
        meta_pbar.set_postfix({'meta_loss': meta_loss.item()})

    return model

def main():
    print("Starting MAML training process")
    reset_cuda_memory()  # Reset CUDA memory before starting the process
    
    # Use the preprocess function from multiModalTrainingV2
    path = "Final_Dataset/Multimodal_final.csv"
    print(f"Loading and preprocessing data from {path}")
    train_data, val_data, test_data = preprocess(path)

    # Prepare datasets
    print("Preparing datasets")
    train_dataset, val_dataset, test_dataset = prepare_datasets(train_data, val_data, test_data)
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize model
    print("Initializing model")
    model = MultiModalModel().to(device)

    # Load pre-trained weights if available
    if os.path.exists("Multimodal_Chckpt/modelEpoch5.pth"):
        print("Loading pre-trained weights")
        model.load_state_dict(torch.load("Multimodal_Chckpt/modelEpoch5.pth", map_location=device))
        print("Pre-trained weights loaded successfully")
    else:
        print("No pre-trained weights found. Starting from scratch.")

    # Perform MAML
    print("Starting MAML outer loop")
    final_model = outer_loop_maml(model, train_dataset, val_dataset, test_dataset)

    # Save final model weights
    final_weights_path = "MAML_Chckpt/final_model_weights.pth"
    torch.save(final_model.state_dict(), final_weights_path)
    print(f"MAML training completed. Final model weights saved to {final_weights_path}")

if __name__ == "__main__":
    main()
