# Computer Vision + LLM Text Generation App

## Overview
This application integrates computer vision (object detection) and a large language model (LLM) to generate meaningful text responses based on both an image and a user-provided prompt.

## Features
- Detects objects in an image using a pre-trained model (Faster R-CNN).
- Generates a coherent text response using a pre-trained LLM (GPT-2).
- Combines image analysis and user prompt for unified output.
- Modular design and robust error handling.

## Setup Instructions

### 1. Clone the repository or copy the files

### 2. Install dependencies
Open a terminal in the project directory and run:

```
pip install torch torchvision pillow transformers
```

### 3. Run the application

```
python main.py
```

## Usage Guide
1. When prompted, enter the path to an image file (e.g., `sample.jpg`).
2. Enter a text prompt (e.g., "Describe the scene for a story.").
3. The app will display detected objects and generate a text response that combines the image analysis and your prompt.

## Notes
- The first run may take longer as models are downloaded.
- Supported image formats: JPEG, PNG, etc.
- Requires Python 3.7 or higher.

## Troubleshooting
- If you see import errors, ensure all dependencies are installed.
- For CUDA acceleration, install the appropriate PyTorch version for your GPU.

## License
This project uses open-source pre-trained models from PyTorch and HuggingFace.
