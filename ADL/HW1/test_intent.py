import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from tqdm import tqdm
import pandas as pd


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_params = {'batch_size': args.batch_size, 'collate_fn': dataset.collate_fn}
    testloader = DataLoader(dataset, **test_params)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = torch.device(args.device)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model = model.to(device)
    model.eval()

    # load weights into model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    prediction = torch.tensor([], device='cuda:0')
    with torch.no_grad():
        for _, data in tqdm(enumerate(testloader)):    
            tokens = data['tokens'].to(device)
            outputs = model(tokens)
            _, indices = torch.max(outputs, 1)
            prediction = torch.cat((prediction, indices))

    # TODO: write prediction to file (args.pred_file)
    pred = []
    for i in range(len(prediction)):
        pred.append([f'test-{i}', dataset.idx2label(prediction[i].item())])
    
    df = pd.DataFrame(pred)
    df.to_csv(args.pred_file, header=["id","intent"], index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
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
