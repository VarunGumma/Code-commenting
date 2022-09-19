import config
import argparse
import torch
from UtilClasses import Tester
from DataWorks import BatchQueue
from Models import DeepComEncoder, AttentionDecoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run the Model")
parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size for the model")
args = parser.parse_args()


batchqueue = BatchQueue(args.batch_size, key="valid")

encoder = DeepComEncoder(inp_dim_code=config.vocab_size_code+1,
                         inp_dim_ast=config.vocab_size_ast+1).to(device)

decoder = AttentionDecoder(inp_dim=config.vocab_size_nl+1).to(device)

model_tester = Tester(encoder=encoder,
                      decoder=decoder,
                      batchqueue=batchqueue,
                      batch_sz=args.batch_size)

model_tester.test()
