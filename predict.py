import torch
import argparse
from utils import *
from model import *
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
device = "cpu"

def virus_embedding(model_name_or_path, fasta_file_path, embedding_file_path):
    covgpt2 = CovGPT2(model_name_or_path)
    autoencoder = torch.load("autoencoder.pkl", map_location=device)
    sentence = get_sequence(fasta_file_path)
    hidden_state = covgpt2.forward(sentence)
    y = autoencoder.forward(hidden_state)
    embedding = flatten_and_pad(y)
    torch.save(embedding, embedding_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-model_name_or_path', dest="model_name_or_path", default='output', help='A path to a *directory* containing a configuration file saved')
    parser.add_argument('-fasta_file_path', dest="fasta_file_path", help='A path to a *directory* containing a fasta file saved')
    parser.add_argument('-embedding_file_path', dest="embedding_file_path", help='Save a embedding file to the *directory*')
    args = parser.parse_args()

    virus_embedding(args.model_name_or_path, args.fasta_file_path, args.embedding_file_path)