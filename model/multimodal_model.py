# models/multimodal_model.py

import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, text_feature_size, face_feature_size, num_classes):
        super(MultimodalModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(text_feature_size + face_feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_features, face_features):
        combined = torch.cat((text_features, face_features), dim=1)
        output = self.classifier(combined)
        return output
