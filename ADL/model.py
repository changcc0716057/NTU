from typing import Dict
from unicodedata import bidirectional

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.gru = torch.nn.GRU(
            input_size=self.embed.weight.shape[1], 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            batch_first = True,
            dropout = self.dropout, 
            bidirectional = self.bidirectional
        )
        self.bn = torch.nn.BatchNorm1d(
            num_features=self.encoder_output_size
        )
        self.fc = torch.nn.Linear(
            in_features=self.encoder_output_size,
            out_features=self.num_class
        )
        self.relu = torch.nn.ReLU()
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size
        #　raise NotImplementedError

    def forward(self, batch) -> torch.Tensor:
        # TODO: implement model forward
        # D = 2 if self.bidirectional else 1
        # h_0 = torch.zeros(D * self.num_layers, batch.shape[0], self.hidden_size)
        # c_0 = torch.zeros(D * self.num_layers, batch.shape[0], self.hidden_size)

        # shape of output: (batch size, sequence length, D * hidden size), D = 2 if bidirection else 1
        batch = self.embed.weight[batch]
        output, _ = self.gru(batch)
        output = self.bn(output[:,-1,:])
        # input of fc: the hidden state of the last word for each batch
        output = self.relu(output)
        output = self.fc(output)
        return output
        # raise NotImplementedError


class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqTagger, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.gru = torch.nn.GRU(
            input_size=self.embed.weight.shape[1], 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            batch_first = True,
            dropout = self.dropout, 
            bidirectional = self.bidirectional
        )
        self.bn = torch.nn.BatchNorm1d(
            num_features=self.encoder_output_size
        )
        self.fc1 = torch.nn.Linear(
            in_features=self.encoder_output_size,
            out_features=self.encoder_output_size
        )   
        self.fc2 = torch.nn.Linear(
            in_features=self.encoder_output_size,
            out_features=self.hidden_size
        )   
        self.fc3 = torch.nn.Linear(
            in_features=self.hidden_size,
            out_features=self.num_class
        )     
        self.prelu1 = torch.nn.PReLU()
        self.prelu2 = torch.nn.PReLU()
        self.relu = torch.nn.ReLU()
        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size
        #　raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        batch = self.embed.weight[batch]
        output, _ = self.gru(batch)
        output = self.bn(output.permute(0,2,1))
        output = self.fc1(output.permute(0,2,1))
        output = self.prelu1(output)
        output = self.fc2(output)
        output = self.prelu2(output)
        output = self.fc3(output)
        return output
