import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Functions for loss calculation
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Evaluation function
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Accuracy calculation function (for classification fine-tuning)
def calc_accuracy_batch(input_batch, label_batch, model, device):
    input_batch = input_batch.to(device)
    label_batch = label_batch.to(device)
    
    with torch.no_grad():
        logits = model(input_batch)
        predictions = logits.argmax(dim=-1)
        correct_predictions = (predictions == label_batch).float().sum()
        accuracy = correct_predictions / label_batch.numel()
    
    return accuracy.item()

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    total_accuracy = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, label_batch) in enumerate(data_loader):
        if i < num_batches:
            accuracy = calc_accuracy_batch(input_batch, label_batch, model, device)
            total_accuracy += accuracy
        else:
            break
    
    return total_accuracy / num_batches

# Function for text generation (for instruction fine-tuning)
def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    context = torch.tensor(tokenizer.encode(start_context), dtype=torch.long).unsqueeze(0).to(device)
    generated_text = start_context
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            if next_token == tokenizer.eos_token_id:
                break
                
            generated_text += tokenizer.decode([next_token])
            context = torch.cat((context, torch.tensor([[next_token]], device=device)), dim=1)
    
    print(generated_text)
    model.train()
    return generated_text

# Classification fine-tuning function
def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    # Initialize lists to track losses and accuracies
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # Track examples
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen

# Instruction fine-tuning function
def train_instruction_model(model, train_loader, val_loader, optimizer, device, num_epochs,
                           eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()  # Track tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

# Function to create a classification fine-tuning dataset
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        tokens = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)
        
        return tokens, torch.tensor(label, dtype=torch.long)

# Function to create an instruction fine-tuning dataset
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format instruction
        formatted_input = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            formatted_input, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).squeeze(0)
        
        # Create target (shifted input_ids)
        target_ids = input_ids.clone()
        target_ids[:-1] = input_ids[1:]
        target_ids[-1] = self.tokenizer.pad_token_id
        
        return input_ids, target_ids

# Main function to run classification fine-tuning
def run_classification_finetuning(model, train_loader, val_loader, device, num_epochs=5, lr=5e-5):
    print("Starting classification fine-tuning...")
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Train model
    start_time = time.time()
    
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    # Plot training metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss During Training')
    
    plt.subplot(1, 2, 2)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy During Training')
    
    plt.tight_layout()
    plt.savefig('classification_finetuning_metrics.png')
    plt.close()
    
    return model

# Main function to run instruction fine-tuning
def run_instruction_finetuning(model, tokenizer, train_loader, val_loader, val_data, device, num_epochs=1, lr=5e-5):
    print("Starting instruction fine-tuning...")
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    # Format sample input for generation during training
    sample_input = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{val_data[0]['instruction']}\n\n### Response:"
    
    # Train model
    start_time = time.time()
    
    train_losses, val_losses, tokens_seen = train_instruction_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=sample_input, tokenizer=tokenizer
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    # Plot training metrics
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Evaluation Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss During Instruction Fine-tuning')
    plt.tight_layout()
    plt.savefig('instruction_finetuning_metrics.png')
    plt.close()
    
    return model

# Example usage:
if __name__ == "__main__":
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "gpt2"  # Using GPT-2 as an example
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Create a sample instruction dataset
    sample_instructions = [
        {
            "instruction": "Convert the active sentence to passive: 'The chef cooks the meal every day.'",
            "response": "The meal is prepared every day by the chef."
        },
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Explain the concept of gravity.",
            "response": "Gravity is a natural force that attracts two objects with mass towards each other."
        }
    ]
    
    # Split into train and validation
    train_data = sample_instructions[:2]
    val_data = sample_instructions[2:]
    
    # Create datasets and data loaders
    train_dataset = InstructionDataset(train_data, tokenizer, max_length=512)
    val_dataset = InstructionDataset(val_data, tokenizer, max_length=512)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Run instruction fine-tuning
    model = run_instruction_finetuning(model, tokenizer, train_loader, val_loader, val_data, device, num_epochs=1)
    
    # Save fine-tuned model
    model.save_pretrained("./instruction_finetuned_model")
    tokenizer.save_pretrained("./instruction_finetuned_model") 