import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_params = {'batch_size': args.batch_size, 'collate_fn': dataset.collate_fn}
    testloader = DataLoader(dataset, **test_params)

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
    model.eval()

    # load weights into model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    prediction = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(testloader)):    
            tokens = data['tokens'].to(device)
            lengths = data['length'].to(device)
            outputs = model(tokens).view(-1, len(tag2idx))
            _, indices = torch.max(outputs, 1)
            indices = indices.view(-1, args.max_len)
            for i in range(indices.shape[0]):
                prediction.append(indices[i][:lengths[i]].tolist())

    # TODO: write prediction to file (args.pred_file)
    pred = []
    for i in range(len(prediction)):
        string = ""
        for j in range(len(prediction[i])):
            string += dataset.idx2label(prediction[i][j]) + " "
        pred.append([f'test-{i}', string.strip()])
    
    df = pd.DataFrame(pred)
    df.to_csv(args.pred_file, header=["id","tags"], index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
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
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.pred_file.parent.mkdir(parents=True, exist_ok=True)
    main(args)