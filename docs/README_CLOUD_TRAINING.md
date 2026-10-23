# Cloud Training for ROP Detection Model

This guide explains how to train the ROP (Retinopathy of Prematurity) detection model on cloud platforms, specifically Kaggle.

## Prerequisites

1. Kaggle account
2. Access to the dataset: https://www.kaggle.com/datasets/jananowakova/retinal-image-dataset-of-infants-and-rop
3. Basic familiarity with Kaggle Notebooks

## Kaggle Training Setup

### 1. Create a New Kaggle Notebook

1. Go to Kaggle.com and sign in to your account
2. Navigate to "Code" section and click "New Notebook"
3. Choose "Notebook" type (not script)

### 2. Configure Notebook Settings

1. In the notebook settings (right sidebar):
   - Enable GPU (required for training)
   - Add the dataset: `jananowakova/retinal-image-dataset-of-infants-and-rop`
   - Make sure internet is enabled

### 3. Upload Files

Upload these files to your Kaggle notebook:
- `train_rop_kaggle.ipynb` (the main training notebook)
- `requirements.txt` (dependency specifications)

### 4. Run the Training

Execute the notebook cells in order:
1. First cell installs dependencies
2. Following cells will load data, create model, train, and evaluate

## Alternative: Direct Kaggle API Upload

If you want to create your own version of the notebook programmatically:

### Install Kaggle CLI
```bash
pip install kaggle
```

### Configure Kaggle Credentials
Place your `kaggle.json` file in `~/.kaggle/kaggle.json` with proper permissions:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Create and Push Notebook
```bash
# Initialize a new kernel
kaggle kernels init -p ./rop-training

# Push the kernel
kaggle kernels push -p ./rop-training

# Run the kernel
kaggle kernels output rop-training
```

## Training Parameters

The notebook uses these default parameters:
- Model: ResNet-50 with ImageNet pretraining
- Input size: 224x224 RGB images
- Batch size: 8
- Epochs: 5 (adjustable)
- Learning rate: 1e-4
- Optimizer: Adam
- Loss function: Binary Cross Entropy

## Expected Results

With the default settings, you should expect:
- Training time: 10-30 minutes for 5 epochs (depending on GPU)
- Accuracy: 80-95% (varies based on data split and randomness)

## Customization Options

You can modify these aspects of the training:

1. **Dataset sampling**: Change `sample_size` parameter to use more/fewer images
2. **Model architecture**: Replace ResNet-50 with other torchvision models
3. **Hyperparameters**: Adjust learning rate, batch size, epochs
4. **Data augmentation**: Modify transforms in the data loader
5. **Preprocessing**: Adjust the image preprocessing pipeline

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size
2. **Dataset not found**: Verify dataset is properly attached to notebook
3. **Import errors**: Ensure all dependencies are installed
4. **Slow training**: Make sure GPU is enabled in notebook settings

### Checking GPU Usage

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Cost Considerations

Kaggle provides free GPU quota:
- Typically 30-40 hours per week for free accounts
- More available with verified accounts
- Premium accounts (Kaggle Pro) offer extended quotas

Monitor your usage at: https://www.kaggle.com/account

## Model Deployment

After training, the model is saved as `rop_detection_model.pth` in the working directory. You can download this file for local use or deploy it as an API service.