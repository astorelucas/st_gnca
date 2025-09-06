
def train_gnca_model(gnca, train_loader, optimizer, criterion, num_epochs, temp_dim, device):

    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            # Batch X shape: torch.Size([32, 10, 358]), Batch y shape: torch.Size([32, 358])
            print(f"Batch X shape: {X_batch.shape}, Batch y shape: {y_batch.shape}")
            
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = gnca.call_model(X_batch)

            print(f"Outputs shape: {outputs.shape}")
            y_batch = y_batch[:,temp_dim:]
            loss = criterion(outputs, y_batch)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            
        # Optionally print the loss for this batch
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)