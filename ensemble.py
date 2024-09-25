import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import CustomDataset
from transform.transform_selector import TransformSelector
from model.model_selector import ModelSelector

class ModelEnsemble:
    def __init__(self, 
                 testdata_dir, 
                 testdata_info_file, 
                 save_result_path, 
                 model_configs=[{'name':'beitv2_large_patch16_224', 'input_size':(224, 224)},
                                {'name':'convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384', 'input_size':(384, 384)},
                                {'name':'deit_base_distilled_patch16_384', 'input_size':(384, 384)},
                                {'name':'eva02_large_patch14_448', 'input_size':(448, 448)}], 

                 batch_size=64,
                 num_classes=500, 
                 pretrained=False,
                 num_workers=6):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testdata_dir = testdata_dir
        self.testdata_info_file = testdata_info_file
        self.save_result_path = save_result_path
        self.model_configs = model_configs
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_workers = num_workers
        self.models = []
        self.transforms = []
        self.test_loader = None

    def prepare_data(self):
        # Load test data information from the CSV file.
        test_info = pd.read_csv(self.testdata_info_file)

        # Set up transformations for test data.
        transform_selector = TransformSelector(transform_type="torchvision")
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
        for config in self.model_configs:
            model_name = config['name']
            input_size = config['input_size']
        
            # Initialize the model using the selected architecture and number of classes.
            model_selector = ModelSelector(
                model_type='timm', 
                num_classes=self.num_classes,
                model_name=model_name, 
                pretrained=self.pretrained
            )
            model = model_selector.get_model()

            # Load the best model checkpoint.
            model_path = os.path.join(self.save_result_path, model_name, "best_model.pt")
            model.load_state_dict(torch.load(model_path, map_location='cpu'))

            # Move the model to the selected device.
            model.to(self.device)
            model.eval()

            # Append to models list for ensembling
            self.models.append(model)

            # Define thr transform for the model-specific input size
            model_transform = transforms.Compose([
                transforms.Resize(input_size),
            ])
            self.transforms.append(model_transform)

    def inference(self):
        predictions = []
        with torch.no_grad():  # Disable gradient computation for inference
            for images in tqdm(self.test_loader):
                ensemble_logits = None

                # Get predictions from each model
                for i, model in enumerate(self.models):
                    model_transform = self.transforms[i]  # Get model-specific transform

                    # Apply the model-specific resizing transform to the batch of images
                    resized_images = torch.stack([model_transform(image) for image in images])

                    # Move resized images to the same device as the model
                    resized_images = resized_images.to(self.device)

                    # Get logits from the current model
                    logits = model(resized_images)
                    logits = F.softmax(logits, dim=1)  # Convert logits to probabilities

                    # Accumulate logits for averaging (ensemble by averaging probabilities)
                    if ensemble_logits is None:
                        ensemble_logits = logits
                    else:
                        ensemble_logits += logits
                
                # Average the logits from all models
                ensemble_logits /= len(self.models)

                # Get the predicted class based on the ensemble output
                preds = ensemble_logits.argmax(dim=1)
                predictions.extend(preds.cpu().detach().numpy())  # Move to CPU and convert to numpy
        
        return predictions

    def save_predictions(self, predictions, test_info):
        # Save the predictions to the test_info dataframe and to a CSV file
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})
        
         # Ensure output path exists and save the results
        output_dir = os.path.join(self.save_result_path, "ensemble")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "output.csv")
        test_info.to_csv(output_file, index=False)

    def run_inference(self):
        # Prepare data
        test_info = self.prepare_data()

        # Prepare model
        self.prepare_model()

        # Perform inference
        predictions = self.inference()

        # Save predictions to CSV
        self.save_predictions(predictions, test_info)
