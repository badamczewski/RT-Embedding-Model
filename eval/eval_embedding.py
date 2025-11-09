import torch
import torch.nn.functional as F
from model import EmbeddingModel, SentencePieceTokenizer

# Load tokenizer
tokenizer = SentencePieceTokenizer(model_path="models/tokenizer.model")
vocab_size = len(tokenizer)

ckpt = torch.load("models/checkpoints/checkpoint_epoch_4.pt", map_location="cpu")

state_dict = ckpt["model_state_dict"]
hidden_dim = state_dict["token_embedding.weight"].shape[1]
embedding_dim = state_dict["projection.3.weight"].shape[0]  # projection.3 is the final Linear layer
vocab_size_from_checkpoint = state_dict["token_embedding.weight"].shape[0]
assert vocab_size == vocab_size_from_checkpoint, "Vocab size mismatch!"

context_window = 16 # Change if needed.

model = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=context_window,
    hidden_dim=hidden_dim,
    is_masked_language=False,  
    use_temperature=False
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded in embedding mode.\n")

pairs = [
    ("I love pizza", "I like pizza"),                # similar
    ("I love pizza", "He drives a car"),             # different
    ("She plays tennis", "He plays tennis"),         # similar
    ("The weather is nice", "It is raining today"),  # moderately related
]

def encode_sentence(text):
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(token_ids) > model.context_window:
        token_ids = token_ids[:model.context_window]
    elif len(token_ids) < model.context_window:
        token_ids += [tokenizer.pad_id] * (model.context_window - len(token_ids))

    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        embedding = model(input_tensor)  # shape: [1, embedding_dim]
    return embedding.squeeze(0)

# Compute cosine similarity between pairs of sentences
for s1, s2 in pairs:
    emb1 = encode_sentence(s1)
    emb2 = encode_sentence(s2)
    cos_sim = F.cosine_similarity(emb1, emb2, dim=0).item()
    print(f"\n{s1!r} <-> {s2!r}")
    print(f"Cosine similarity: {cos_sim:.4f}")
