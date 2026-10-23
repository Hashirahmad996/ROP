# Cloud Training Guide for ROP Detection Model

This guide explains how to train your ROP detection model on various cloud platforms: Kaggle, Google Colab, and Hugging Face.

## Prerequisites

### 1. Dataset Preparation
- Organize your dataset in a cloud-friendly format
- Upload to the platform or connect to external storage
- Ensure proper directory structure matching your training script expectations

### 2. Code Modifications Needed
Your current script requires these changes for cloud compatibility:

#### Path Adjustments
Change hardcoded paths to configurable parameters:
```python
# Current (hardcoded)
dataset_dir = r"C:\Users\Hashir Ahmad\Downloads\AI Agents\ROP\dataset\images_stack_without_captions\images_stack_without_captions"

# Recommended (configurable)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='/kaggle/input/your-dataset-folder', help='Path to dataset')
args = parser.parse_args()
dataset_dir = args.dataset_path
```

#### Add Progress Tracking
For long training sessions, add progress indicators:
```python
# Add tqdm for progress bars
from tqdm import tqdm

# Wrap your data loader
for i, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
    # training code
```

#### Save Model Checkpoints
Save intermediate checkpoints to avoid losing progress:
```python
# Save checkpoint every few epochs
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }, f'rop_model_checkpoint_{epoch}.pth')
```

## Platform-Specific Setup

### Kaggle

#### 1. Dataset Upload
- Compress your dataset into a ZIP file
- Upload via Kaggle Datasets → "New Dataset"
- Choose "Files" and upload your ZIP file
- Make dataset public or private as needed

#### 2. Kernel Setup
Create a new Notebook with these settings:
- GPU accelerator (required for training)
- Add your uploaded dataset as input data

#### 3. Installation Commands
Add to the first cell:
```python
!pip install torch torchvision opencv-python-headless scikit-learn tqdm
```

#### 4. Path Configuration
In Kaggle, datasets are mounted at `/kaggle/input/DATASET_NAME/`

#### 5. Submit to Competition (if applicable)
- Use "Save Version" → "Save & Run All (Commit)"
- Choose appropriate runtime settings (GPU, hours limit)

### Google Colab

#### 1. Dataset Options
Three ways to access data:
a) Upload directly to Colab (temporary, session-based)
b) Connect Google Drive and mount:
```python
from google.colab import drive
drive.mount('/content/drive')
```
c) Download from external source:
```python
!wget https://your-data-source.com/dataset.zip
!unzip dataset.zip
```

#### 2. Environment Setup
```python
# Install dependencies
!pip install torch torchvision opencv-python-headless scikit-learn tqdm

# GPU check
import torch
print(torch.cuda.is_available())
```

#### 3. Runtime Configuration
- Runtime → Change runtime type → Hardware accelerator: GPU
- May need Colab Pro for extended training sessions

### Hugging Face

#### 1. Repository Setup
- Create a new repository on Hugging Face
- Clone locally and push your code:
```bash
git clone https://huggingface.co/YOUR_USERNAME/YOUR_REPO_NAME
cd YOUR_REPO_NAME
# Add your files
git add .
git commit -m "Initial commit"
git push
```

#### 2. Dataset Integration
Upload dataset to Hugging Face Datasets or use Spaces storage

#### 3. Training Compute
Options for training:
a) Local training with HF transformers
b) Spaces with GPU (limited time)
c) Training Compute (paid service)

## Code Improvements for Cloud Training

### 1. Enhanced Training Script
Here's an improved version of your training loop:

```python
def train_model(model, train_loader, test_loader, device, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Validation phase
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).squeeze()
                preds = (outputs > 0.5).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())
        
        acc = accuracy_score(all_targets, all_preds)
        print(f"Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} - Acc: {acc:.4f}")
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_rop_model.pth")
            print(f"New best model saved with accuracy: {acc:.4f}")
    
    return model
```

### 2. Configuration Management
Create a config file or use argparse for better parameter management:

```python
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='ROP Detection Training')
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate')
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224), 
                        help='Image size (height, width)')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save outputs')
    return parser.parse_args()
```

## Resource Requirements

### Hardware
- GPU recommended (NVIDIA Tesla, V100, T4, etc.)
- Minimum 8GB VRAM for your ResNet-50 model
- Storage space for dataset (~1-10GB depending on image count)

### Time Estimation
Based on your current script:
- Small subset (40 training images): ~5-10 minutes
- Full dataset: Several hours depending on size
- Recommendation: Start with a small subset to verify everything works

## Cost Considerations

### Kaggle
- Free tier: 30 hours/week of GPU time
- GPU specifications vary (Tesla P100, T4, etc.)

### Google Colab
- Free tier: Limited GPU access
- Colab Pro: ~$10/month for extended access
- Colab Pro+: ~$50/month for priority access

### Hugging Face
- Free tier: Limited compute time
- Paid plans for extended training

## Next Steps

1. Prepare your dataset for upload (organize, compress)
2. Modify your training script with the improvements above
3. Test locally with a small subset
4. Deploy to your chosen cloud platform
5. Monitor training progress and adjust hyperparameters as needed