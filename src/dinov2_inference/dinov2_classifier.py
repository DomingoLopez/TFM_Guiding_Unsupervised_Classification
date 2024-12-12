import http
import json
import sys
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
from loguru import logger
from tqdm import tqdm


class DinoV2Classifier:
    """
    A classifier that uses DinoV2 embeddings.
    Embeddings are either loaded from cache or generated on-the-fly.
    Includes methods for training, prediction, and accuracy evaluation.
    """
    def __init__(self, 
                 model_name="small", 
                 model_path=None, 
                 images=None, 
                 num_classes=5,
                 disable_cache = False, 
                 verbose=False):
        
        # Initializing model
        json_sizes_path = Path(__file__).resolve().parent / "json/dinov2_sizes.json"
        with open(json_sizes_path,'r') as model_sizes:
            self.model_name = json.load(model_sizes).get(model_name)

        self.model_folder = "facebookresearch/dinov2" if model_path is None else model_path
        self.model_source = "github" if model_path is None else "local"
        self.disable_cache = disable_cache
        
        # Validate image list
        if not isinstance(images, list):
            raise TypeError(f"Expected 'images' to be a list, but got {type(images).__name__} instead.")
        if len(images) < 1:
            raise ValueError("The 'images' list must contain at least one image.")
        self.images = images  

        # Setup logging
        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")


        # CUDA Available or not
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info("Using GPU for inference")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for inference")


        try:
            logger.info(f"loading {self.model_name=} from {self.model_folder=}")
            self.model = torch.hub.load(
                self.model_folder,
                self.model_name,
                source=self.model_source,
            )
        except FileNotFoundError:
            logger.error(f"load model failed. please check if {self.model_folder=} exists")
            sys.exit(1)
        except http.client.RemoteDisconnected:
            logger.error(
                "connect to github is reset. maybe set --model-path to $HOME/.cache/torch/hub/facebookresearch_dinov2_main ?"
            )
            sys.exit(1)

        # Model to device (GPU or CPU)
        self.model.to(self.device)
        # Setup model in eval mode.
        self.model.eval()

        # Construct image tranforms
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Generate cache folder if cache up
        if not self.disable_cache:
            # cache_root_folder = Path(
            #     appdirs.user_cache_dir(appname="dinov2_inference", appauthor="domi")
            # )
            parent_root = Path(__file__).resolve().parent  # Un nivel hacia arriba desde src/
            cache_root_folder = parent_root / "cache"
            cache_root_folder.mkdir(parents=True, exist_ok=True)
            self.embeddings_cache_path = cache_root_folder / (
                "embeddings_" + self.model_name + "_" + str(len(self.images)) + ".pkl"
            )
            logger.debug(f"{cache_root_folder=}, {self.embeddings_cache_path=}")
        

        # Initialize the classifier model
        json_emb_sizes_path = Path(__file__).resolve().parent / "json/dinov2_embeddings_sizes.json"
        with open(json_emb_sizes_path,'r') as model_emb_sizes:
            self.embedding_dim = json.load(model_emb_sizes).get(model_name)

        self.num_classes = num_classes
        self.model = nn.Linear(self.embedding_dim, self.num_classes).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)





    def __is_cache_the_same(self):
        """
        Check if the number of images loaded are the same that we have in cache
        in order to load or not cache
        """
        if self.embeddings_cache_path is not None:
            total_cache = pickle.load(
                open(str(self.embeddings_cache_path), "rb")
            )
            return len(total_cache) == len(self.images)
        else:
            return False


    def load_embeddings(self):
        """
        Load embeddings from cache or generate them if cache does not exist.
        """
        if self.embeddings_cache_path and Path(self.embeddings_cache_path).exists():
            logger.info(f"Loading embeddings from cache: {self.embeddings_cache_path}")
            with open(self.embeddings_cache_path, "rb") as f:
                embeddings_data = pickle.load(f)
        else:
            embeddings_data = None

        return embeddings_data



    def train(self, embeddings, labels, epochs=10, batch_size=32):
        """
        Train the classifier on provided embeddings and labels.
        Args:
            embeddings: Precomputed embeddings.
            labels: Corresponding labels for embeddings.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        dataset = EmbeddingDataset(embeddings, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for batch_embeddings, batch_labels in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                batch_embeddings, batch_labels = batch_embeddings.to(self.device), batch_labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_embeddings)
                loss = self.criterion(outputs, batch_labels)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")




    def predict(self, embeddings):
        """
        Predict the classes for given embeddings.
        Args:
            embeddings: Precomputed embeddings to classify.
        Returns:
            List of predicted class indices.
        """
        self.model.eval()
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(embeddings_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def evaluate_accuracy(self, embeddings, labels):
        """
        Evaluate the accuracy of the classifier on provided embeddings and labels.
        Args:
            embeddings: Precomputed embeddings.
            labels: Ground truth labels.
        Returns:
            Accuracy score as a percentage.
        """
        predictions = self.predict(embeddings)
        accuracy = accuracy_score(labels, predictions)
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy * 100


class EmbeddingDataset(Dataset):
    """
    A PyTorch Dataset for embeddings and labels.
    """
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
