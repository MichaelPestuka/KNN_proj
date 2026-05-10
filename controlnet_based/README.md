# Controlnet based approach

This part of the project implements an autoregressive, diffusion-based game engine for generating continuous Super Mario Bros gameplay. The system uses a Transformer to predict future game states based on user actions, a Segmentation Decoder to construct control maps, and a Stable Diffusion + ControlNet pipeline to render the final gameplay frames in real-time.

Below is a brief manual describing the purpose and functionality of each core script.


### `data_loader.py`
Dataset management and pre-processing.
- Handles the loading of game episodes and parses JSON metadata.
- Implements the `MarioDataset` PyTorch class, managing sliding windows of historical frames.
- Converts raw game actions (left, right, A, B) into multi-hot encoded vectors.
- Manages transformations, image normalizations, and padding for episode boundaries.

### `inference_engine.py`
Real-time gameplay inference.
Provides the `InferenceEngine` class, an optimized wrapper designed to be queried in real-time.
- Maintains an internal history buffer of recent states and actions.
- Exposes a simple interface: input the current player action, and it outputs the next generated RGB frame.
- Automatically handles the autoregressive shifting of the internal buffer.

### `pipeline.py`
Core model orchestration and training loop.
Defines the `DiffusionPipeline` which links all components together (Transformer $\rightarrow$ Segmentation Decoder $\rightarrow$ ControlNet $\rightarrow$ Stable Diffusion).
- Implements the `Trainer` class responsible for the model's training loop, including multi-component optimization, gradient checkpointing, and unfreezing specific modules.


### `seg_precompute.py`
Offline segmentation map generation.
A heavily parallelized script that pre-computes ground-truth semantic segmentation maps for the training data.
- Uses OpenCV to analyze raw game frames, applying HSV color thresholding and texture matching (Z-indexed) to classify pixels into specific classes (sky, ground, Mario, block, enemy, cloud, pipe).
- Saves the processed maps for faster dataloading during the actual training phase.

### `transformer.py`
Implements the `GameStateTransformer`, which takes an embedded history of actions, physics, and visual/latent states to predict the next game state.
- Contains the `SegmentationDecoder`, a convolutional network that decodes the Transformer's output into a semantic segmentation map.
- Defines the `LatentCompressor`, used to encode spatial maps into lower-dimensional latent representations.