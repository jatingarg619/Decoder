import os
import torch
from huggingface_hub import HfApi, create_repo, upload_file
from train_optimized import GPT, GPTConfig  # Import your model architecture
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_to_hub(
    checkpoint_path="checkpoints/latest_model.pt",
    repo_name="your-username/shakespeare-gpt",  # Change this to your desired repo name
    token=None  # Your HF token
):
    if token is None:
        token = os.getenv('HF_TOKEN')
        if token is None:
            raise ValueError("Please provide a Hugging Face token or set HF_TOKEN environment variable")
    
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Create or get repository
        repo_url = create_repo(
            repo_name,
            private=False,
            token=token,
            exist_ok=True
        )
        print(f"Repository URL: {repo_url}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Initialize model with the same config
        model = GPT(GPTConfig(vocab_size=checkpoint['model_state_dict']['lm_head.weight'].shape[0]))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Save model in PyTorch format
        torch.save(model.state_dict(), "pytorch_model.bin")
        
        # Create config file
        config = {
            "architectures": ["GPT"],
            "model_type": "gpt",
            "vocab_size": model.config.vocab_size,
            "n_layer": model.config.n_layer,
            "n_head": model.config.n_head,
            "n_embd": model.config.n_embd,
            "block_size": model.config.block_size,
            "dropout": model.config.dropout,
            "bias": model.config.bias
        }
        
        # Save config
        import json
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Create README
        readme_content = f"""
# Shakespeare Text Generator

This is a decoder-only transformer model trained on Shakespeare's text, designed to generate text in Shakespeare's distinctive style. The model has been trained to achieve a loss < 0.099999, ensuring high-quality text generation.

## Model Architecture

The model is a GPT-style decoder-only transformer with the following specifications:
- **Total Parameters**: {sum(p.numel() for p in model.parameters())/1e6:.2f}M
- **Layers**: {model.config.n_layer} transformer blocks
- **Attention Heads**: {model.config.n_head} heads per block
- **Embedding Dimension**: {model.config.n_embd}
- **Context Length**: {model.config.block_size} tokens
- **Vocabulary**: Character-level tokenization
- **Activation**: GELU
- **Layer Normalization**: Pre-norm configuration
- **Attention**: Flash Attention for efficient computation

## Training Details

- **Training Data**: Shakespeare's complete works
- **Training Objective**: Next character prediction
- **Optimizer**: AdamW
  - Learning Rate: 3e-4
  - Weight Decay: 0.1
  - Beta1: 0.9
  - Beta2: 0.95
- **Learning Rate Schedule**: Cosine decay with warmup
- **Gradient Clipping**: 1.0
- **Achieved Loss**: < 0.099999

## Sample Outputs

### Example 1: Famous Hamlet Soliloquy
![Sample Output 1](images/sample1.png)

In this example, the model continues the famous "To be, or not to be" soliloquy with parameters:
- Temperature: 0.8
- Top-k: 50
- Max New Tokens: 100

### Example 2: Casual Conversation in Shakespeare Style
![Sample Output 2](images/sample2.png)

Here, the model transforms a modern casual greeting into Shakespeare's style with parameters:
- Temperature: 0.1 (more deterministic)
- Top-k: 1 (most likely token)
- Max New Tokens: 100

## Usage

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("{repo_name}")
```

For a user-friendly interface, visit our Hugging Face Space (link coming soon).

## Limitations

- Character-level tokenization means the model works best with standard English characters
- The context length is limited to {model.config.block_size} characters
- The model may occasionally generate anachronistic content
- Best results are achieved with prompts in a Shakespearean style

## Citation

```bibtex
@misc{{shakespeare-decoder,
  author = {{Jatin Garg}},
  title = {{Shakespeare Text Generator}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```
"""
        
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        # Upload files
        files_to_upload = [
            "pytorch_model.bin",
            "config.json",
            "README.md",
            "input.txt",
            "images/sample1.png",
            "images/sample2.png"
        ]
        
        # Create images directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        # Copy input.txt to current directory if it's in CodeFiles
        if not os.path.exists("input.txt") and os.path.exists("CodeFiles/input.txt"):
            import shutil
            shutil.copy2("CodeFiles/input.txt", "input.txt")
        
        for file in files_to_upload:
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_name,
                token=token
            )
        
        print("Successfully uploaded model to Hugging Face Hub!")
        print(f"View your model at: https://huggingface.co/{repo_name}")
        
        # Clean up local files
        for file in files_to_upload:
            os.remove(file)
            
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {str(e)}")
        raise

if __name__ == "__main__":
    # Use token from environment variable
    upload_to_hub(
        checkpoint_path="checkpoints/latest_model.pt",
        repo_name="jatingocodeo/shakespeare-decoder"  # Replace with your repository name
    ) 