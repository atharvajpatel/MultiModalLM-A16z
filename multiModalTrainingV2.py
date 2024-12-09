import os
import torch
import pytorch_lightning as pl
import csv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer, ViTModel, ViTImageProcessor
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set CUDA device and limit memory usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # Limit GPU memory usage to 95%
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_memory = int(total_memory * 0.95)
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.cuda.empty_cache()
    print(f"GPU memory limited to {target_memory / (1024**3):.2f} GB")
print(f"Using device: {device}")

# Path to dataset
path = "Final_Dataset/Multimodal_final.csv"

# Preprocessing function
def preprocess(path):
    print("Starting data preprocessing...")
    df = pd.read_csv(path)
    print(f"Loaded dataset with {len(df)} samples.")

    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    image_processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')

    def process_data(df):
        texts = list(df['Caption'])
        image_paths = list(df['Image Path'])
        labels = list(df['LABEL'])
        
        print(f"Processing {len(texts)} samples...")
        tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        images = []
        for img_path in tqdm(image_paths, desc="Processing images"):
            image = Image.open(img_path).convert('RGB')
            processed_image = image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
            images.append(processed_image)
        images = torch.stack(images)

        return tokenized_texts, images, torch.tensor(labels).long()

    train_data = process_data(train_df)
    val_data = process_data(val_df)
    test_data = process_data(test_df)

    print("Data preprocessing completed.")
    return train_data, val_data, test_data

# Dataset class
class MultimodalDataset(Dataset):
    def __init__(self, texts, images, labels):
        self.texts = texts
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.texts['input_ids'][idx],
            'attention_mask': self.texts['attention_mask'][idx],
            'image': self.images[idx],
            'label': self.labels[idx]
        }

# MLP module
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super().__init__()
        self.query = nn.Linear(dim_q, dim_v)
        self.key = nn.Linear(dim_k, dim_v)
        self.value = nn.Linear(dim_k, dim_v)
        self.scale = dim_v ** -0.5

    def forward(self, x, y):
        q = self.query(x)
        k = self.key(y)
        v = self.value(y)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return out

class MultiModalModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-5):
        super().__init__()
        print("Initializing MultiModalModel...")
        
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        
        self.distilbert_output_dim = self.distilbert.config.hidden_size  # 768 for distilbert-base
        self.vit_output_dim = self.vit.config.hidden_size  # 1024 for vit-large

        # Cross-attention layers
        self.text_to_image_attn = CrossAttention(self.distilbert_output_dim, self.vit_output_dim, 512)
        self.image_to_text_attn = CrossAttention(self.vit_output_dim, self.distilbert_output_dim, 512)

        # MLPs for each modality
        self.text_mlp = MLP(self.distilbert_output_dim + 512, 512, 256)
        self.image_mlp = MLP(self.vit_output_dim + 512, 512, 256)

        # Final classifier
        self.classifier = nn.Linear(512, num_classes)
        
        self.learning_rate = learning_rate
        print("MultiModalModel initialized.")
        
    def forward(self, input_ids, attention_mask, image):
        # Process text
        text_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_pooled = text_output.mean(dim=1)
        
        # Process image
        image_output = self.vit(pixel_values=image).last_hidden_state
        image_pooled = image_output.mean(dim=1)

        # Cross-attention
        text_attended = self.text_to_image_attn(text_pooled, image_pooled)
        image_attended = self.image_to_text_attn(image_pooled, text_pooled)

        # Concatenate original and attended features
        text_features = torch.cat([text_pooled, text_attended], dim=-1)
        image_features = torch.cat([image_pooled, image_attended], dim=-1)

        # Pass through MLPs
        text_features = self.text_mlp(text_features)
        image_features = self.image_mlp(image_features)

        # Combine features
        combined_features = torch.cat([text_features, image_features], dim=-1)

        # Classification
        logits = self.classifier(combined_features)
        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, image, labels = batch['input_ids'], batch['attention_mask'], batch['image'], batch['label']
        logits = self(input_ids, attention_mask, image)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('train_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_accuracy', acc, prog_bar=True)
        return {'loss': loss, 'train_accuracy': acc}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, image, labels = batch['input_ids'], batch['attention_mask'], batch['image'], batch['label']
        logits = self(input_ids, attention_mask, image)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('val_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_accuracy', acc, prog_bar=True)
        return {'val_loss': loss, 'val_accuracy': acc}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, image, labels = batch['input_ids'], batch['attention_mask'], batch['image'], batch['label']
        logits = self(input_ids, attention_mask, image)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('test_loss', loss)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('test_accuracy', acc, prog_bar=True)
        return {'test_loss': loss, 'test_accuracy': acc}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

# Training the model
if __name__ == "__main__":
    print("Starting multimodal training process...")
    train_data, val_data, test_data = preprocess(path)
    
    print("Creating datasets...")
    train_dataset = MultimodalDataset(*train_data)
    val_dataset = MultimodalDataset(*val_data)
    test_dataset = MultimodalDataset(*test_data)

    print("Setting up data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)

    print("Initializing model...")
    model = MultiModalModel()

    print("Setting up trainer...")
    trainer = pl.Trainer(
        max_epochs=5,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        accumulate_grad_batches=8,  # Gradient accumulation
        precision=16  # Use mixed precision training
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Saving metrics...")
    metrics_file = 'metrics.csv'
    metrics = trainer.logged_metrics
    with open(metrics_file, mode='w', newline='') as csv_file:
        fieldnames = ['epoch', 'train_accuracy', 'val_accuracy']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(trainer.current_epoch):
            writer.writerow({'epoch': epoch, 'train_accuracy': metrics['train_accuracy'], 'val_accuracy': metrics['val_accuracy']})
    print('Metrics saved.')

    print("Starting testing...")
    trainer.test(model, test_loader)
    
    print("Saving model weights...")
    torch.save(model.state_dict(), f'Multimodal_Chckpt/modelEpoch{trainer.current_epoch}.pth')
    print('Model weights saved.')

    print("Multimodal training process completed.")