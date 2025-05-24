"""Used for model build with Roberta pre-trained transformer"""

from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers.modeling_outputs import (
    SequenceClassifierOutput,   
)   
          
class RobertaForSequenceClassificationSig(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # self.sigmoid = torch.sigmoid
        self.sigmoid = torch.nn.functional.hardsigmoid # HardSigmoid
        self.threshold = torch.nn.Threshold(-1,-1)

    def customHardSigmoid(self, x):
        return torch.nn.functional.hardsigmoid(3*x)
        
    def customActivation(self, x):
        # 使用 Tanh 激活函数并进行线性变换
        return 0.5 * (torch.tanh(x) + 1)  # 变换到 [0, 1] 范围    
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        
        # ret = super.forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ret = super(RobertaForSequenceClassificationSig, self).forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
        ret.logits = self.sigmoid(ret.logits) # Uncomment to use any sigmoid
        ret.logits = torch.relu(ret.logits) # Uncomment to use ReLu w threshold
        ret.logits = -self.threshold(-ret.logits) # Uncomment to use ReLu w threshold
        # .logits = self.customActivation(ret.logits)  # 使用自定义激活函数
        return ret
    