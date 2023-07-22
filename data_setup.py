import zipfile
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def unzip_file_dir(source: str, destination: str):
    '''
    This function takes a zipped file path and a destinatination path that the file will be unzipped to.

    parameters
    -----------
    source: [str] -> The directory path the zipped folder is located in
    destination: [str] -> The directory the zipped folder should be unzipped in
    '''

    with zipfile.ZipFile(source, 'r') as zip_ref:
        zip_ref.extractall(destination)

    prompt= f'[INFO]: your file has been unzipped to {destination} directory'

    return prompt


def walkthrough_dir(dir_path: str):
  '''
  walks through a dir path in order to return the summary of its content
  
  parameter
  ---------
  dir_path: The directory to be walked through
  '''

  for dirpath, dirname,filenames in os.walk(dir_path):
    print (f'there are {len(dirname)} directories and {len(filenames)} images in {dirpath}')



def dataloader_function(train_dir: Path,
                        test_dir: Path,
                        val_dir: Path,
                        transform: transforms.Compose,
                        batchsize,
                        num_worker,
                        test_val_transform= transforms.Compose([
                           transforms.Resize(size= (224,224)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])):
        '''
        This function takes in an image path and return the transformed image data in batches
        parameters
        -----------
        train_dir: The train data directory
        test_dir: The test data directory
        val_dir: The val data directory
        transform: Transforms for the data
        batchsize: The batchsize the data should be trained on
        num_worker: No of workers
        '''

        train_data= datasets.ImageFolder(train_dir, transform)
        test_data= datasets.ImageFolder(test_dir, transform= test_val_transform)
        val_data= datasets.ImageFolder(val_dir, test_val_transform)
        class_names= train_data.classes

        train_dataloader= DataLoader(dataset= train_data,
                                     batch_size= batchsize,
                                     shuffle= True,
                                     num_workers=num_worker)
        test_dataloader= DataLoader(dataset= test_data,
                                    batch_size= batchsize, 
                                    shuffle= False, 
                                    num_workers=num_worker)
        val_dataloader= DataLoader(dataset= val_data,
                                   batch_size= batchsize, 
                                   shuffle= False, 
                                   num_workers=num_worker)
        

        return train_dataloader, test_dataloader, val_dataloader, class_names
