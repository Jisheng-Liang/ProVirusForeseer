import torch

def get_sequence(file_path):
    with open(file_path, 'r', errors='ignore') as f:
        lines = f.readlines()
        sentence = "<|endoftext|>" + lines[1].strip()
        return sentence

def flatten_and_pad(tensor):
    max_len = 46848
    embedding = tensor.squeeze().view(1, -1)
    embedding_len = tensor.shape[1]
    if embedding_len < max_len:
        embedding = torch.nn.functional.pad(tensor, (0, max_len-embedding_len), mode='constant', value=0)
    elif embedding_len == max_len:
        pass
    else:
        embedding = embedding_len[:, :max_len]
    return embedding

# python xxx.py -i seq.fasta -o res.pt