import torch
from torch import nn 
from transformers import GPT2Model, GPT2Tokenizer

class Classifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str, compression_ratio:int):
        super(Classifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.pooling = nn.AdaptiveMaxPool1d(compression_ratio)
        self.fc1 = nn.Linear(compression_ratio*max_seq_len*7, num_classes)
       

        
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
          
            gpt_out_pooling= self.pooling(gpt_out)
            
            batch_size = gpt_out_pooling.shape[0]
            concatenated_sub_tensors.append(gpt_out_pooling)
        
        result = torch.cat(concatenated_sub_tensors, dim=1)
        batch_size = result.shape[0]
        linear_output = self.fc1(result.view(batch_size,-1))
        return linear_output