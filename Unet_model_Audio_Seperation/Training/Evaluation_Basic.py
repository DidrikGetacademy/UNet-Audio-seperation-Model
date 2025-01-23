
import torch 
#VALIDATION FUNCTION
def Validate_epoch(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_combined_loss = 0
    val_mask_loss = 0
    val_hybrid_loss = 0

    with torch.no_grad():
        for val_batch_idx, (inputs, targets) in enumerate(val_loader):
      
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
          
            predicted_mask = model(inputs.to(device, non_blocking=True))

            if predicted_mask.size() != targets.size():
                raise ValueError(f"Validation shape mismatch: predicted_mask={predicted_mask.size()}, targets={targets.size()}")

           
            combined_loss, mask_loss, hybrid_loss = criterion(
                predicted_mask.to(device), 
                inputs.to(device), 
                targets.to(device)
            )

    
            val_combined_loss += combined_loss.item()
            val_mask_loss += mask_loss.item()
            val_hybrid_loss += hybrid_loss.item()


    avg_combined_loss = val_combined_loss / len(val_loader)
    avg_mask_loss = val_mask_loss / len(val_loader)
    avg_hybrid_loss = val_hybrid_loss / len(val_loader)

    return avg_combined_loss, avg_mask_loss, avg_hybrid_loss
