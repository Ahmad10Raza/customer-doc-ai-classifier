import os
import argparse
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
import numpy as np
from evaluate import load

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT for Customer PO classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./vit-customer-model", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()

    # ============= CUDA SETUP ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= LABELS ===================
    labels_names = sorted(os.listdir(args.data_dir))
    label2id = {label: i for i, label in enumerate(labels_names)}
    id2label = {i: label for i, label in enumerate(labels_names)}

    # ================= CUSTOM DATASET ==================
    class CustomImageDataset(Dataset):
        def __init__(self, data_dir, label2id, transform=None):
            self.images = []
            self.labels = []
            self.transform = transform
            for label in sorted(os.listdir(data_dir)):
                folder = os.path.join(data_dir, label)
                for file in os.listdir(folder):
                    if file.lower().endswith((".jpg",".jpeg",".png")):
                        self.images.append(os.path.join(folder, file))
                        self.labels.append(label2id[label])
        def __len__(self):
            return len(self.images)
        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return {"pixel_values": img, "labels": self.labels[idx]}

    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    def transform(img):
        return processor(img, return_tensors="pt")["pixel_values"].squeeze()

    full_dataset = CustomImageDataset(args.data_dir, label2id, transform=transform)
    n_val = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    def collate_fn(batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.tensor([x["labels"] for x in batch])
        return {"pixel_values": pixel_values, "labels": labels}

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(labels_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        no_cuda=not torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    accuracy = load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model_save_path = os.path.join(args.output_dir, "final")
    trainer.save_model(model_save_path)
    processor.save_pretrained(model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    main()
