import torch
from torch import nn
from transformers import BertModel



class BertFor2Classification(nn.Module):
    def __init__(self, config_bert):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',
                                              output_hidden_states=True)
        self.dropout = nn.Dropout(config_bert.dropout)
        self.classifier = nn.Linear(config_bert.hidden_size, 1)#
        for param in self.bert.parameters():
            param.requires_grad = True
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                train = True
                ):
        # print("num_choices",num_choices)
        # print(num_choices)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        d_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(d_pooled_output)
        if train:
            return logits,d_pooled_output
        else:
            return logits.view(-1, 5)