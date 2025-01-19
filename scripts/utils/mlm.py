from transformers import PreTrainedTokenizer

from dnazen.ngram import NgramEncoder

def make_mlm_dataset(
    tokenizer: PreTrainedTokenizer,
    file: str,
):
    pass


def _make_mlm_dataset_from_fasta(
    tokenizer: PreTrainedTokenizer,
    file: str,
    max_seq_len: int,
    ngram_encoder: NgramEncoder
):
    from Bio import SeqIO
    
    with open(file, "r") as f:
        dna_seq_list = SeqIO.read(f, "fasta")
    
    