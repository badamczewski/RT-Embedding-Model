import torch
from model import EmbeddingModel, SentencePieceTokenizer

# Load tokenizer
tokenizer = SentencePieceTokenizer(model_path="models/tokenizer.model")

# Load model
model = EmbeddingModel(
    vocab_size=len(tokenizer),
    embedding_dim=256,
    context_window=16,
    hidden_dim=512,
    is_masked_language=True,  # MLM mode
    use_temperature=False
)

# Load checkpoint
ckpt = torch.load("models/checkpoints/checkpoint_epoch_1.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Encode sentence with a MASK
text = "This is a [MASK]" 
token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
print("Input token IDs:", token_ids)

# Convert to tensor shape (1, seq_len)
input_tensor = torch.tensor([token_ids], dtype=torch.long)

# Run model
with torch.no_grad():
    logits = model(input_tensor)

# Find top prediction at mask position
mask_id = tokenizer.mask_id
mask_index = (input_tensor[0] == mask_id).nonzero(as_tuple=True)[0].item()

# Get top-k predictions for that position
topk = torch.topk(logits[0, mask_index], k=5)
pred_ids = topk.indices.tolist()
pred_tokens = [tokenizer.decode([i]) for i in pred_ids]

print("\nTop-5 predictions for the [MASK] token:")
for i, (tok, score) in enumerate(zip(pred_tokens, topk.values.tolist())):
    print(f"{i+1}. {tok} ({score:.3f})")
