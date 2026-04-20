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

        # Load pretrained DistilBERT encoder
        self.bert = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # A slightly deeper head is more stable across URLs, free text, and HTML.
        self.pre_classifier = nn.Linear(768, 256)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.35)
        self.fc = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean pooling works better than a single first-token embedding
        # for heterogeneous text such as URLs, email text, and HTML.
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # Classification head
        x = self.pre_classifier(pooled)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits.squeeze(1)
