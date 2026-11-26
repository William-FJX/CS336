"""Run once from assignment1-basics after placing the TinyStories text files in data/full/."""
import ast
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import tiktoken

from cs336_basics.tokenizer import PAT


END_OF_TEXT = "<|endoftext|>"


def story_chunks(source):
    """Split immediately after a story delimiter so no BPE token spans two chunks."""
    pending = ""
    while block := source.read(2 ** 20):
        pending += block
        boundary = pending.rfind(END_OF_TEXT)
        if boundary >= 0:
            end = boundary + len(END_OF_TEXT)
            yield pending[:end]
            pending = pending[end:]
    if pending:
        yield pending


def prepare_split(text_path, encoder):
    output_path = text_path.with_suffix(".bin")
    with TemporaryDirectory(dir=output_path.parent) as temp_dir:
        staged = Path(temp_dir) / output_path.name
        with text_path.open(encoding="utf-8") as source, staged.open("wb") as destination:
            for chunk in story_chunks(source):
                ids = encoder.encode_to_numpy(chunk, allowed_special={END_OF_TEXT})
                ids.astype("<u2").tofile(destination)
        total = staged.stat().st_size // 2
        os.replace(staged, output_path)
    print(f"{output_path}: {total:,} tokens", flush=True)


if __name__ == "__main__":
    # In this trained vocabulary IDs are in BPE merge order.
    vocab = json.loads(Path("output/tinystories_vocab.json").read_text())
    ranks = {ast.literal_eval(token): int(token_id) for token_id, token in vocab.items()}
    if max(ranks.values()) >= 2 ** 16:
        raise ValueError("TinyStories token IDs must fit in uint16.")
    special_id = ranks.pop(END_OF_TEXT.encode())
    encoder = tiktoken.Encoding(name="tinystories", pat_str=PAT, mergeable_ranks=ranks,
                                special_tokens={END_OF_TEXT: special_id})
    for split in ("valid", "train"):
        prepare_split(Path(f"data/full/TinyStoriesV2-GPT4-{split}.txt"), encoder)
