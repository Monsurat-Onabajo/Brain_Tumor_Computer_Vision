from tqdm.auto import tqdm
import torch

# Training model
def train_loop(train_dataloader, model, loss_fn, optimizer, device):
  loss= 0
  accuracy=0

  model.train()
  for batchsize, (X, y) in enumerate(train_dataloader):
    X, y= X.to(device), y.to(device)
    y_logits= model(X)
    train_loss= loss_fn(y_logits, y)
    loss += train_loss.item()
    y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    accuracy += (y_pred_class == y).sum().item()/len(y_logits)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
  loss /= len(train_dataloader)
  accuracy /= len(train_dataloader)

  return loss, accuracy


def test_loop(test_dataloader, model, loss_fn, device):
  test_loss= 0
  test_accuracy=0
  model.eval()
  with torch.inference_mode():
    for batch_size, (X, y) in enumerate(test_dataloader):
      X, y= X.to(device), y.to(device)
      y_logits= model(X)
      test_loss += (loss_fn(y_logits, y)).item()
      test_pred= torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
      test_pred_labels = y_logits.argmax(dim=1)
      test_accuracy += ((test_pred_labels == y).sum().item()/len(test_pred_labels))


  test_loss /= len(test_dataloader)
  test_accuracy /= len(test_dataloader)

  return test_loss, test_accuracy


def training_loop(train_dataloader, test_dataloader, epochs, loss_fn, device, optimizer, model):

  results= {
      "train_loss": [],
      "train_accuracy": [],
      "test_loss": [],
      "test_accuracy": []
      }

  epochs= epochs
  for item in tqdm(range(epochs)):
    train_loss, train_accuracy= train_loop(
        train_dataloader=train_dataloader, model= model,
        loss_fn= loss_fn, optimizer=optimizer, device= device,
    )

    test_loss, test_accuracy= test_loop(
        test_dataloader= test_dataloader, model= model,
        device= device, loss_fn= loss_fn
    )

    results['train_loss'].append(train_loss)
    results['train_accuracy'].append(train_accuracy)
    results['test_loss'].append(test_loss)
    results['test_accuracy'].append(test_accuracy)

    print (f'train accuracy: {train_accuracy} | train_loss: {train_loss} | test accuracy: {test_accuracy} | test loss: {test_loss}')

  return results
