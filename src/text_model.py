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

        # Classification head
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        # Pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Classification head
        x = self.dropout(cls_embedding)
        logits = self.fc(x)

        return logits.squeeze(1)