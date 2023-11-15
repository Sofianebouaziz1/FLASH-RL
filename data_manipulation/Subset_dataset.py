import torch
from torch.utils.data import Dataset

class Subset_dataset(Dataset):
    """
    Sous-ensemble d'un ensemble de données aux indices spécifiés.

    Arguments :
        dataset (Ensemble de données) : Le jeu de données complet
        indices (séquence) : Indices dans l'ensemble entier sélectionnés pour le sous-ensemble
        labels(sequence) : cibles comme requis pour les indices. seront de la même longueur que les indices
    """
    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels
        
    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)