import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM


class CovGPT2():
    def __init__(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    def forward(self, sentence):
        input_ids = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_state = outputs.hidden_states[-1]
        return hidden_state

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y