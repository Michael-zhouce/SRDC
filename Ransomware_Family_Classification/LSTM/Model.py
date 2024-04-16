import torch
from torch import nn 
from transformers import GPT2Model, GPT2Tokenizer

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.lstm = nn.LSTM(hidden_size = compression_ratio,input_size = 768, num_layers =1,
                            batch_first=True)
        self.fc1 = nn.Linear(compression_ratio*7, num_classes)
        
    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        input_ids = torch.split(input_id, 1, dim=1)
        masks = torch.split(mask, 1, dim=1)
        concatenated_sub_tensors = []
        for sub_input_id, sub_mask in zip(input_ids, masks):
            
            sub_input_id = sub_input_id.squeeze(1)
            
            gpt_out, _ = self.gpt2model(input_ids=sub_input_id, attention_mask=sub_mask, return_dict=False)

            gpt_out_pooling, _ = self.lstm(gpt_out)
            concatenated_sub_tensors.append(gpt_out_pooling[:, -1, :])
        result = torch.cat(concatenated_sub_tensors, dim=1)
        linear_output = self.fc1(result)
        return linear_output