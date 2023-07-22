import torch
def eval_model(model, dataloader, loss_fn, device):
  test_loss= 0
  test_acc=0
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      X,y= X.to(device), y.to(device)
      y_pred= model(X)
      test_loss += loss_fn(y_pred, y)
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      test_acc += (y_pred_class == y).sum().item()/len(y_pred)
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return {
        'model name': model.__class__.__name__,
        'accuracy': test_acc,
        'loss': test_loss
    }

      
     
