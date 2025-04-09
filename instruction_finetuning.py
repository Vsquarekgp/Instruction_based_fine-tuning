import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import urllib
import ssl

# Function to download and load instruction data
def download_and_load_file(file_path, url):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url, context=ssl_context) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data

# Function to format input in Alpaca format
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

# Loss calculation functions
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

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Text generation function
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

# Instruction dataset class
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format instruction input
        formatted_input = format_input(item)
        formatted_output = f"\n\n### Response:\n{item['output']}"
        formatted_text = formatted_input + formatted_output
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            formatted_text, 
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

# Training function
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
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

if __name__ == "__main__":
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model_name = "gpt2"  # Using GPT-2 as an example
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Download and load instruction dataset
    print("Loading instruction dataset...")
    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    print(f"Number of entries: {len(data)}")
    
    # Split into train and validation (90% train, 10% validation)
    train_size = int(0.9 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    print(f"Training on {len(train_data)} examples")
    print(f"Validating on {len(val_data)} examples")
    
    # Create datasets and data loaders
    train_dataset = InstructionDataset(train_data, tokenizer, max_length=512)
    val_dataset = InstructionDataset(val_data, tokenizer, max_length=512)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Move model to device
    model.to(device)
    
    # Calculate initial loss
    print("\nCalculating initial loss...")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
    print(f"Initial training loss: {train_loss:.3f}")
    print(f"Initial validation loss: {val_loss:.3f}")
    
    # Training setup
    print("\nStarting training...")
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 1
    
    # Start training
    start_time = time.time()
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\nTraining completed in {execution_time_minutes:.2f} minutes.")
    
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
    
    # Save fine-tuned model
    print("\nSaving model...")
    model.save_pretrained("./instruction_finetuned_model")
    tokenizer.save_pretrained("./instruction_finetuned_model")
    print("Model saved to ./instruction_finetuned_model") 