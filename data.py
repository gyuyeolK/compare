"""
Data loading.

Two modes:
  1. 'fineweb': stream HuggingFaceFW/fineweb-edu (sample-10BT) and tokenize
     on the fly with the GPT-2 tokenizer. This is what the Dion paper uses
     (well, FineWeb-Edu).
  2. 'synthetic': random integers. Useful for quick optimizer-time benchmarks
     and for environments without internet.

Both modes return iterators of (input_ids, target_ids) tensors with shape
(batch_size, seq_len), where target_ids = input_ids shifted by one.
"""

import torch


# ----------------------------- synthetic ----------------------------------- 

def synthetic_loader(batch_size, seq_len, vocab_size, device, n_batches=10**9, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(n_batches):
        ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1),
                            generator=g)
        ids = ids.to(device, non_blocking=True)
        yield ids[:, :-1], ids[:, 1:]


# ----------------------------- FineWeb (streaming) -------------------------

def fineweb_loader(batch_size, seq_len, device, dataset="HuggingFaceFW/fineweb-edu",
                   subset="sample-10BT", split="train", tokenizer_name="gpt2",
                   seed=0):
    """Streaming token loader.

    Concatenates tokenized documents with the EOT separator into a flat
    buffer, then yields (B, T+1) windows.

    Requires `datasets` and `transformers` to be installed and internet access.
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    # the model_max_length warning is harmless here (we never feed the
    # tokenizer's model with these tokens), but silence it anyway:
    tok.model_max_length = 10**9
    eot = tok.eos_token_id
    ds = load_dataset(dataset, name=subset, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    buf = torch.empty(0, dtype=torch.long)
    it = iter(ds)
    chunk = batch_size * (seq_len + 1)  # number of tokens needed per batch

    while True:
        # refill buffer until we have enough tokens for one batch
        while buf.numel() < chunk:
            try:
                doc = next(it)
            except StopIteration:
                return
            ids = tok.encode(doc["text"]) + [eot]
            buf = torch.cat([buf, torch.tensor(ids, dtype=torch.long)])

        batch = buf[:chunk].view(batch_size, seq_len + 1).contiguous()
        buf = buf[chunk:]
        batch = batch.to(device, non_blocking=True)
        yield batch[:, :-1], batch[:, 1:]


def get_loader(kind, batch_size, seq_len, vocab_size, device, **kwargs):
    if kind == "fineweb":
        return fineweb_loader(batch_size, seq_len, device, **kwargs)
    elif kind == "synthetic":
        return synthetic_loader(batch_size, seq_len, vocab_size, device, **kwargs)
    else:
        raise ValueError(f"unknown loader kind: {kind}")
