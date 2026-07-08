"""Stream OpenWebText + GPT-2 BPE → train/val .bin (uint16), capped at MAX_TRAIN_TOKENS.
No 54GB raw cache (streaming). val = first VAL_DOCS docs. GPT-2 tiktoken (same as wikitext)."""
import os, numpy as np, tiktoken, time
from datasets import load_dataset

OUT = os.path.dirname(os.path.abspath(__file__))
MAX_TRAIN_TOKENS = 2_500_000_000   # ~2.5B (Chinchilla-optimal for 124M; >= training budget → no reuse)
VAL_DOCS = 4000
BATCH = 1024
enc = tiktoken.get_encoding("gpt2"); eot = enc.eot_token

def main():
    ds = load_dataset("openwebtext", streaming=True, split="train", trust_remote_code=True)
    ftr = open(os.path.join(OUT,"train.bin"),"wb"); fva = open(os.path.join(OUT,"val.bin"),"wb")
    buf=[]; n_docs=n_tr=n_va=0; t0=time.time()
    for ex in ds:
        buf.append(ex["text"])
        if len(buf) >= BATCH:
            for ids in enc.encode_ordinary_batch(buf):
                ids.append(eot); arr=np.array(ids,dtype=np.uint16)
                if n_docs < VAL_DOCS: fva.write(arr.tobytes()); n_va+=len(ids)
                else:
                    if n_tr >= MAX_TRAIN_TOKENS: buf=[]; break
                    ftr.write(arr.tobytes()); n_tr+=len(ids)
                n_docs+=1
            buf=[]
            if n_tr and n_docs % 100000 == 0:
                el=time.time()-t0
                print(f"  {n_docs} docs, train={n_tr/1e9:.2f}B/{MAX_TRAIN_TOKENS/1e9:.1f}B, {el:.0f}s ({n_tr/max(el,1)/1e6:.1f}M/s)", flush=True)
            if n_tr >= MAX_TRAIN_TOKENS: break
    ftr.close(); fva.close()
    print(f"DONE: train={n_tr/1e9:.3f}B val={n_va/1e6:.1f}M tokens ({n_docs} docs)", flush=True)

if __name__ == "__main__": main()
