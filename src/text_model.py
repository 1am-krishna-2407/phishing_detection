import torch
import torch.nn as nn
from transformers import AutoModel


class TextPhishingModel(nn.Module):
    """
    DistilBERT-based text phishing detection model
    Outputs a single logit for binary classification
    """

    def __init__(self):
        super(TextPhishingModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )

        self.pre_classifier = nn.Linear(768, 256)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        x = self.pre_classifier(pooled)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits.squeeze(1)
