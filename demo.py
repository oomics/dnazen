# %% setups
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from time import time

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True, use_fast=True
)
assert tokenizer.is_fast

for tok in [[128, 723], [37, 1417], [55, 584], [550, 128], [45, 701]]:
    out = tokenizer.decode(tok)
    print(out)

# %%

# MAX_SEQ_LEN = 10_000_000

# beg = time()
# res = []
# print("reading dna...")
# # max_res = 100; i = 0
# total_len = 0
# dna_seq_str_list = []
# for seq_record in SeqIO.parse("datasets/hg38.fa", "fasta"):
#     dna_seq = str(seq_record.seq).upper().replace("N", "")
#     dna_seq_str_list.append(dna_seq)
#     # print(f"tokenizing... (sample seq: {dna_seq[:20]}; seq_len={len(dna_seq)})")
#     # inputs = tokenizer(dna_seq)
#     # res.append(inputs["input_ids"])
#     # total_len += len(inputs["input_ids"])
#     # print(dna_seq[:10])

# print("Tokenizing...")
# results = tokenizer(dna_seq_str_list)["input_ids"]
# print(results[0])

# end = time()
# print(end-beg)

# DEST_DIR = "datasets/hg38-tokenized.txt"
# with open(DEST_DIR, "w") as f:
#     print("writing to file: ", DEST_DIR)
#     for seq in results:
#         t = ":".join([str(tok) for tok in seq])
#         f.write(t + "\n")

# %% count ngram
from time import time
from _ngram import NgramFinderConfig, DnaNgramFinder

DEST_DIR = "datasets/hg38-tokenized.txt"
results = []
with open(DEST_DIR, "r") as f:
    print("reading from file: ", DEST_DIR)
    step = 0
    for line in f.readlines():
        nums = line.split(":")
        results.append([int(num) for num in nums])
        step += 1
        if step == 400:
            break

# count num tokens
num_tokens = 0
for res in results:
    num_tokens += len(res)
print(num_tokens)
# exit()


config = NgramFinderConfig()

config.min_ngram_freq = 5
config.min_ngram_len = 1
config.max_ngram_len = 6
config.min_pmi = 1
config.min_token_count = 5
config.num_workers = 64

finder = DnaNgramFinder(config)
beg = time()
finder.find_ngrams_batched(results)
end = time()

res = finder.get_ngram_list([])
print(res[:100])

with open("datasets/ngram.txt", "w") as f:
    for r in res:
        r_freq = r.pop()

        # r_str = ":".join([str(_) for _ in r])
        r_str = tokenizer.decode(r)
        f.write(r_str + "," + str(r_freq) + "\n")

# print("input length=", total_len)
print("length of res=", len(res))
print("time taken:", end - beg)
# %%
