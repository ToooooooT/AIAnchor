import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch_size', type=int, default=64, help="batch size")
    parse.add_argument('--epoch', default=3, type=int, help="epoch")
    parse.add_argument('--lr', default=2e-5, type=float, help="lr")
    parse.add_argument('--val_freq', default=1, help="validation frequency")
    args = parse.parse_args()
    return args


def validation(model, val_loader):
    model.eval()
    val_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()

    average_val_loss = val_loss / len(val_loader)
    accuracy = correct_predictions / len(val_dataset)

    print(f'Validation Loss: {average_val_loss}')
    print(f'Validation Accuracy: {accuracy}')
    return accuracy


if __name__ == '__main__':
    args = parse_args()
    # Load your own dataset
    # Assuming you have a CSV file with columns 'text' and 'label'
    df = pd.read_csv('dataset/data.csv')

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    LR = args.lr

    # Initialize RoBERTa tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=45)  # Adjust num_labels based on your classification task

    # Prepare custom datasets and dataloaders
    train_dataset = CustomDataset(train_df['text'].values, train_df['label'].values, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = CustomDataset(val_df['text'].values, val_df['label'].values, tokenizer, MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # tensorboard
    writer = SummaryWriter(f'logs/{BATCH_SIZE}_{LR}')

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    max_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(train_loader)
        print(f'Training Loss: {average_loss}')

        writer.add_scalar('Loss', total_loss, global_step=epoch + 1)

        # Validation loop
        if (epoch + 1) % args.val_freq == 0:
            acc = validation(model, val_loader)
            if acc > max_acc:
                torch.save(model, f'weights/model_{LR}_{BATCH_SIZE}.pth')
                max_acc = acc
            writer.add_scalar('Validation accuracy', acc, global_step=epoch + 1)
