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
# Shakespeare GPT Model

This is a GPT model trained on Shakespeare's text. The model has the following specifications:

- Vocabulary Size: {model.config.vocab_size}
- Number of Layers: {model.config.n_layer}
- Number of Heads: {model.config.n_head}
- Embedding Dimension: {model.config.n_embd}
- Context Length: {model.config.block_size}
- Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M

## Usage

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("{repo_name}")
```

## Training
This model was trained on Shakespeare's text with a target loss of < 0.099999.
        """
        
        with open("README.md", "w") as f:
            f.write(readme_content)
        
        # Upload files
        files_to_upload = [
            "pytorch_model.bin",
            "config.json",
            "README.md"
        ]
        
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
        repo_name="jatingocodeo/shakespeare-decoder"
    ) 