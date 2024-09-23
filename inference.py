import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.dataset import CustomDataset
from transform.transform_selector import TransformSelector
from model.model_selector import ModelSelector
import wandb

class ModelInference:
    def __init__(self, 
                 testdata_dir, 
                 testdata_info_file, 
                 save_result_path, 
                 model_name='resnet18', 
                 batch_size=64,
                 num_classes=500, 
                 pretrained=False,
                 num_workers=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testdata_dir = testdata_dir
        self.testdata_info_file = testdata_info_file
        self.save_result_path = save_result_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_workers = num_workers
        self.model = None
        self.test_loader = None

    def prepare_data(self):
        # Load test data information from the CSV file.
        test_info = pd.read_csv(self.testdata_info_file)

        # Set up transformations for test data.
        transform_selector = TransformSelector(transform_type="customed_torchvision")
        test_transform = transform_selector.get_transform(is_train=False)

        # Create dataset instance for test data.
        test_dataset = CustomDataset(
            root_dir=self.testdata_dir,
            info_df=test_info,
            transform=test_transform,
            is_inference=True
        )

        # Set up data loader for batching.
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False,
            num_workers=self.num_workers
        )
        return test_info

    def prepare_model(self):
        # Initialize the model using the selected architecture and number of classes.
        model_selector = ModelSelector(
            model_type='timm', 
            num_classes=self.num_classes,
            model_name=self.model_name, 
            pretrained=self.pretrained
        )
        self.model = model_selector.get_model()

        # Load the best model checkpoint.
        model_path = os.path.join(self.save_result_path, self.model.model_name,"best_model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        # Move the model to the selected device.
        self.model.to(self.device)
        self.model.eval()

    def inference(self):
        predictions = []
        with torch.no_grad():  # Disable gradient computation for inference
            for images in tqdm(self.test_loader):
                # Move images to the same device as the model
                images = images.to(self.device)

                # Perform prediction
                logits = self.model(images)
                logits = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                # Store predictions
                predictions.extend(preds.cpu().detach().numpy())  # Move to CPU and convert to numpy

        return predictions

    def save_predictions(self, predictions, test_info):
        # Save the predictions to the test_info dataframe and to a CSV file
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        test_info.to_csv(f"/data/ephemeral/home/results/{self.model.model_name}/output.csv", index=False)
        wandb.log({"test_predictions": wandb.Table(dataframe=test_info)})
        wandb.finish()

    def run_inference(self):
        # Prepare data
        test_info = self.prepare_data()

        # Prepare model
        self.prepare_model()

        # Perform inference
        predictions = self.inference()

        # Save predictions to CSV
        self.save_predictions(predictions, test_info)
