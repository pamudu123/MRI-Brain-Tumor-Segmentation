import torch
import os

def train_classifier_one_epoch(model, train_dataloader, val_dataloader, optimizer, criterion, device, 
                               best_val_acc, model_save_path="best_classifier.pth"):
    """
    Trains the classification model for one epoch, evaluates on validation set,
    and saves the best model based on validation accuracy.
    
    Returns:
        tuple: (train_loss, train_acc, val_loss, val_acc, updated_best_val_acc)
    """
    # Training phase
    model.train()
    train_total_loss = 0
    train_correct_predictions = 0
    train_total_samples = 0
    
    for data in train_dataloader:
        if data is None: continue
        
        # Handle batches where some samples might be None (filtered out)
        if any(sample is None for sample in data.values()):
            continue
            
        images = data['classifier_img'].to(device)
        labels = data['classifier_label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total_samples += labels.size(0)
        train_correct_predictions += (predicted == labels).sum().item()
    
    train_avg_loss = train_total_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
    train_accuracy = train_correct_predictions / train_total_samples if train_total_samples > 0 else 0
    
    # Validation phase
    model.eval()
    val_total_loss = 0
    val_correct_predictions = 0
    val_total_samples = 0
    
    with torch.no_grad():
        for data in val_dataloader:
            if data is None: continue
            
            # Handle batches where some samples might be None (filtered out)
            if any(sample is None for sample in data.values()):
                continue
                
            images = data['classifier_img'].to(device)
            labels = data['classifier_label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total_samples += labels.size(0)
            val_correct_predictions += (predicted == labels).sum().item()
    
    val_avg_loss = val_total_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
    val_accuracy = val_correct_predictions / val_total_samples if val_total_samples > 0 else 0
    
    # Save best model based on validation accuracy
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'val_loss': val_avg_loss
        }, model_save_path)
        print(f"New best model saved! Validation accuracy: {val_accuracy:.4f}")
    
    print(f"Classifier Training -> Loss: {train_avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    print(f"Classifier Validation -> Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    
    return train_avg_loss, train_accuracy, val_avg_loss, val_accuracy, best_val_acc

def evaluate_classifier(model, dataloader, criterion, device):
    """
    Evaluates the classification model on a given dataset.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in dataloader:
            if data is None: continue
            images = data['classifier_img'].to(device)
            labels = data['classifier_label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return avg_loss, accuracy

def load_best_classifier_model(model, model_path="best_classifier.pth", device='cpu'):
    """
    Loads the best saved classifier model.
    
    Args:
        model: The model instance to load weights into
        model_path: Path to the saved model
        device: Device to load the model on
    
    Returns:
        dict: Dictionary containing model info (best_val_acc, val_loss)
    """
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with validation accuracy: {checkpoint['best_val_acc']:.4f}")
        return {
            'best_val_acc': checkpoint['best_val_acc'],
            'val_loss': checkpoint['val_loss']
        }
    else:
        print(f"No saved model found at {model_path}")
        return {'best_val_acc': 0.0, 'val_loss': float('inf')}