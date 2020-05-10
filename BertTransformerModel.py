import torch
from transformers import BertModel, BertTokenizer

class BertTransformerModel(torch.nn.Module):

    def __init__(self):
        super(BertTransformerModel, self).__init__()

        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.hidden_size = self.encoder.config.hidden_size

        # TO DO Implement an actual decoder.
        self.decoder = torch.nn.Linear(self.hidden_size, self.hidden_size)

        self.init_weights()

    # Initializes the decoder weights (only). The encoder weights are pre-trained.
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.decoder.weight)

    def forward(self, input):
        # Tokenize the input and run it through BERT.
        tokenized_input = torch.tensor([self.tokenizer.encode(input, add_special_tokens=True)])
        with torch.no_grad():
            output = self.encoder(tokenized_input)[0]

        # output has shape (# samples, # tokens, BERT state size)

        # TO DO Apply the decoder here

        return output
