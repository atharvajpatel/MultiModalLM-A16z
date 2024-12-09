import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertModel, BertTokenizer, ViTModel
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set CUDA device if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to dataset
path = "Final_Dataset/Multimodal_final.csv"

# Preprocessing function
def preprocess(path):
    # Load dataset
    df = pd.read_csv(path)

    # Train/Test/Dev split (60/20/20)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Define BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define image converter
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Tokenize text and process images
    def process_data(df):
        texts = list(df['Caption'])
        image_paths = list(df['Image Path'])
        labels = list(df['LABEL'])
        
        tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        images = [image_transform(Image.open(img_path).convert('RGB')) for img_path in image_paths]
        images = torch.stack(images)

        return tokenized_texts, images, torch.tensor(labels).long()

    train_data = process_data(train_df)
    val_data = process_data(val_df)
    test_data = process_data(test_df)

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

# Model definition
class MultiModalModel(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-4):
        super().__init__()
        
        # Pre-trained models
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        
        # Linear layer for classification
        self.classifier = nn.Linear(1536, num_classes)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        
    def forward(self, input_ids, attention_mask, image):
        # Pass text through BERT
        text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_pooled = text_output.mean(dim=1)  # Apply mean pooling along the sequence dimension
        
        # Pass image through ViT
        image_output = self.vit(pixel_values=image).last_hidden_state
        image_pooled = image_output.mean(dim=1)  # Apply mean pooling along the sequence dimension

        # Concatenate pooled representations
        combined_representation = torch.cat((text_pooled, image_pooled), dim=-1)

        # Classification
        logits = self.classifier(combined_representation)
        return logits
        
        # Classification
        logits = self.classifier(cross_attn_output.mean(dim=0))
        return logits

    def training_step(self, batch, batch_idx):
        from tqdm import tqdm
        print(f"Starting training step {batch_idx + 1}")
        input_ids, attention_mask, image, labels = batch['input_ids'], batch['attention_mask'], batch['image'], batch['label']
        logits = self(input_ids, attention_mask, image)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        print(f"Starting validation step {batch_idx + 1}")
        input_ids, attention_mask, image, labels = batch['input_ids'], batch['attention_mask'], batch['image'], batch['label']
        logits = self(input_ids, attention_mask, image)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, image, labels = batch['input_ids'], batch['attention_mask'], batch['image'], batch['label']
        logits = self(input_ids, attention_mask, image)
        loss = nn.CrossEntropyLoss()(logits, labels)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

# Training the model
if __name__ == "__main__":
    # Preprocess the data
    train_data, val_data, test_data = preprocess(path)
    print('Data preprocessing completed.')
    
    # Create datasets
    train_dataset = MultimodalDataset(*train_data)
    val_dataset = MultimodalDataset(*val_data)
    test_dataset = MultimodalDataset(*test_data)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Model
    model = MultiModalModel()

    # Training
    trainer = pl.Trainer(
        max_epochs=5,
        devices=1 if torch.cuda.is_available() else 0,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        accumulate_grad_batches=4,
    )
    trainer.fit(model, train_loader, val_loader)
    print('Training completed.')
    

    # Testing
    trainer.test(model, test_loader)
    print('Testing completed.')
    print('Starting Testing...')
    trainer.test(model, test_loader)

    # Inference function
def inference(model, data_loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(data_loader, desc='Inference Progress'):
        input_ids, attention_mask, image, labels = batch['input_ids'], batch['attention_mask'], batch['image'], batch['label']
        with torch.no_grad():
            logits = model(input_ids, attention_mask, image)
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Create heatmap
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    