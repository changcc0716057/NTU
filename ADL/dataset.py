from cgitb import text
from typing import List, Dict

from torch.utils.data import Dataset
import torch

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn

        # Transform texts to tokens, encode tokens, and padding to max length.
        texts = [instance['text'].split() for instance in samples]
        encode_texts = self.vocab.encode_batch(texts, self.max_len)
        # If the samples are from testing data, then simply set the value to None.
        # Set the encoding of None to be - 1. 
        intents = [instance.get('intent', 'None') for instance in samples]
        if (intents[0] == 'None'):
            encode_intents = [-1] * len(intents)
        else:
            encode_intents = [self.label2idx(intent) for intent in intents]

        return {
            'tokens': torch.tensor(encode_texts),
            'labels': torch.tensor(encode_intents)
        }
        # raise NotImplementedError

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    def collate_fn(self, samples):
        ignore_idx = -100
        # TODO: implement collate_fn
        # Encode tokens, and padding to max length.
        texts = [instance['tokens'] for instance in samples]
        length = [len(instance['tokens']) for instance in samples]
        encode_texts = self.vocab.encode_batch(texts, self.max_len)
        # If the samples are from testing data, then simply set the value to None.
        # Set the encoding of None to be - 1. 

        if "tags" in samples[0]:
            encode_tags = [[self.label2idx(tag) for tag in instance['tags']] for instance in samples]
            for i in range(len(encode_tags)):
                encode_tags[i] = encode_tags[i] + [ignore_idx] * (self.max_len - len(encode_tags[i]))
        else:
            encode_tags = [ignore_idx for j in range(len(encode_texts[0])) for i in range((len(encode_texts)))]
                                                                                          
        return {
            'tokens': torch.tensor(encode_texts),
            'labels': torch.tensor(encode_tags),
            'length': torch.tensor(length)
        }
