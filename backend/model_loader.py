import os
import torch
import torch.nn as nn
import torchvision.models as models


def get_model_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "training", "models", "gastric_resnet50.pth")


def get_fusion_model_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, "training", "models", "gastric_fusion.pth")


def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model_path = get_model_path()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def load_feature_extractor():
    model = load_model()
    modules = list(model.children())[:-1]
    feature_extractor = nn.Sequential(*modules)
    feature_extractor.eval()
    return feature_extractor


class ClinicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, x):
        return self.net(x)


class GenomicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clinical_net = ClinicalMLP()
        self.genomic_net = GenomicMLP()
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 8 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 8)
        )

    def forward(self, img_features, clinical_data, genomic_data):
        clinical_features = self.clinical_net(clinical_data)
        genomic_features = self.genomic_net(genomic_data)
        combined = torch.cat([img_features, clinical_features, genomic_features], dim=1)
        return self.classifier(combined)


def load_fusion_model():
    fusion = FusionModel()
    fusion_path = get_fusion_model_path()
    if os.path.exists(fusion_path):
        fusion.load_state_dict(torch.load(fusion_path, map_location=torch.device("cpu")))
    fusion.eval()
    return fusion