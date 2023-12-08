 Learning Rate (LR) Scheduler to adjust the LR during training. Models often benefit from this technique once learning stagnates, and you get better results. We will go over the different methods we can use and I'll show some code examples that apply the scheduler.

 Adjusting the learning rate (LR) during training is crucial for optimizing neural network performance. There are various types of LR schedulers used in deep learning. Here are a few common ones:

1. **StepLR**: This scheduler decreases the learning rate by a factor gamma every step_size epochs.
   
   ```python
   from torch.optim.lr_scheduler import StepLR
   
   # Create an optimizer (e.g., SGD)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
   
   # Create a scheduler
   scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
   
   # Inside the training loop
   for epoch in range(num_epochs):
       # Train your model
       # ...
       # Update the learning rate
       scheduler.step()
   ```

2. **ReduceLROnPlateau**: It reduces the LR when a certain metric (e.g., validation loss) stops improving.

   ```python
   from torch.optim.lr_scheduler import ReduceLROnPlateau
   
   # Create an optimizer (e.g., Adam)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   # Create a scheduler
   scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
   
   # Inside the training loop
   for epoch in range(num_epochs):
       # Train your model
       # ...
       # Compute validation loss
       val_loss = validate(model, val_loader)  # Function to compute validation loss
       
       # Update the learning rate based on validation loss
       scheduler.step(val_loss)
   ```

3. **CosineAnnealingLR**: This scheduler applies a cosine annealing schedule to the learning rate.

   ```python
   from torch.optim.lr_scheduler import CosineAnnealingLR
   
   # Create an optimizer (e.g., Adam)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   # Create a scheduler
   scheduler = CosineAnnealingLR(optimizer, T_max=10)
   
   # Inside the training loop
   for epoch in range(num_epochs):
       # Train your model
       # ...
       # Update the learning rate
       scheduler.step()
   ```

These schedulers can be adjusted with different parameters based on your specific problem and dataset. Experimentation with LR schedulers and monitoring the model's performance is key to finding the best learning rate strategy for your task.

 Documentation:
 https://pytorch.org/docs/stable/optim.html
