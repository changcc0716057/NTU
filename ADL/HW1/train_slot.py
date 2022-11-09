from base64 import encode
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    train_params = {'batch_size': args.batch_size, 'collate_fn': datasets[TRAIN].collate_fn}
    dev_params = {'batch_size': args.batch_size, 'collate_fn': datasets[DEV].collate_fn}
    trainloader = DataLoader(datasets[TRAIN], **train_params)
    devloader = DataLoader(datasets[DEV], **dev_params)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    device = torch.device(args.device)
    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        num_class=len(tag2idx)
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    model.train()
    for epoch in epoch_pbar:
        for _, data in tqdm(enumerate(trainloader)):
        # TODO: Training loop - iterate over train dataloader and update model weights
            tokens = data['tokens'].to(device)
            labels = data['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(tokens).view(-1, len(tag2idx))
            loss = loss_fn(outputs, labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.3, norm_type=2)

            optimizer.step()

        n_sample = 0
        n_correct = 0
        # TODO: Evaluation loop - calculate accuracy and save model weights
        for _, data in enumerate(devloader):
            tokens = data['tokens'].to(device)
            labels = data['labels'].to(device)
            lengths = data['length'].to(device)
            outputs = model(tokens).view(-1, len(tag2idx))
            _, indices = torch.max(outputs, 1)
            indices = indices.view(-1, args.max_len)
            n_sample += indices.shape[0]
            for i in range(indices.shape[0]):
                n_correct += torch.equal(indices[i][:lengths[i]], labels[i][:lengths[i]])

        print(f'[epoch {epoch+1}], valid acc: {n_correct/n_sample}')
        ckptname = str(epoch) + '.pt'
        torch.save(model.state_dict(), args.ckpt_dir / ckptname)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=80)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)