# From-Scratch LSTM Implementation with PyTorch Lightning
This project implements a Long Short-Term Memory (LSTM) network from its fundamental components using PyTorch and the PyTorch Lightning framework. As a successor to the from-scratch vanilla RNN, this model is designed to overcome the vanishing gradient problem and capture long-range dependencies in sequential data. The model is trained on a simple, synthetic dataset to verify its ability to learn patterns over a sequence.

## Project Overview
The primary goal of this project is to deconstruct the LSTM architecture and implement its core gating mechanisms (forget, input, and output gates) manually. By using PyTorch Lightning, the project also demonstrates best practices for organizing deep learning code, separating the model's logic from the training boilerplate.

This implementation showcases a deep, first-principles understanding of:

* LSTM Architecture: The internal workings of the cell state and the three primary gates.

* Modern Training Frameworks: The use of LightningModule to structure the model, training loop, and optimizer configuration.

* Experiment Tracking: Integration with TensorBoard for logging and visualizing training metrics like loss.


## Methodology
The model is a single-layer LSTM cell that processes an input sequence one time step at a time. An LSTM maintains both a short-term memory (the hidden state, h_t) and a long-term memory (the cell state, C_t). The flow of information is controlled by three gates:

* The forget gate: ( I think this should called a remember gate) because it determines how much of the long term memory should be kept based on the existing short term memory, and the input at that stage
* The input gate: How much of the current short term memory should we add to the long term memory
* The output gate: How much of the new long term memory should become the new short term memory passed on to the next stage. 

https://colah.github.io/posts/2015-08-Understanding-LSTMs/ (this blog post goes into the depth of it. )


## Training with PyTorch Lightning
The entire model is encapsulated in a LightningModule, which organizes the code into distinct, clean sections:

__init__(): Initializes all learnable parameters (weights and biases for each gate).

forward(): Defines the logic for processing a full input sequence, iterating through each time step and updating the LSTM's internal states.

training_step(): Calculates the loss (Mean Squared Error) for a given batch and logs it for monitoring.

configure_optimizers(): Specifies the optimizer (Adam) to be used for training.

The training process is handled by the lightning.Trainer, which automates the training loop, gradient updates, and checkpointing.


## Results 
I used the model to predict values given a simple time series.
Here are some visualization using tensor board to show how the predictions improve as I add more epochs to the model's training. 

<img width="1000" height="507" alt="image" src="https://github.com/user-attachments/assets/1af78183-2533-45e1-a5c2-888f54a03315" />
<img width="942" height="508" alt="image" src="https://github.com/user-attachments/assets/420a655d-e7b7-4042-bdff-04a50d0c76be" />
<img width="1007" height="507" alt="image" src="https://github.com/user-attachments/assets/265080c5-5c21-4612-8112-5b409caae9d1" />
<img width="857" height="484" alt="image" src="https://github.com/user-attachments/assets/0f907aaf-aea4-4626-abaa-aa9b83abd95a" />




