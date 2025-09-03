
def train_gnca_model(gnca, train_loader, optimizer, criterion, device):

    total_loss = 0
    for batch in train_loader:

        optimizer.zero_grad()

        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        outputs = gnca.call_model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)