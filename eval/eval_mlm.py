import torch
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

# Load model
model = EmbeddingModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    context_window=16,
    hidden_dim=hidden_dim,
    is_masked_language=True,  # MLM mode
    use_temperature=False
)

# Load checkpoint
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


for text in [
    "She has won multiple Swiss [MASK] and currently holds four",
    "They went to the [MASK] yesterday",
    "I love eating [MASK] pizza",
]:

    # Encode sentence with a MASK
    token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)

    # Convert to tensor shape (1, seq_len)
    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    print("Input text:", text)
    print("Input token IDs:", token_ids)


    # Run model
    with torch.no_grad():
        logits = model(input_tensor)

    # Find top prediction at mask position
    mask_id = tokenizer.mask_id
    mask_index = (input_tensor[0] == mask_id).nonzero(as_tuple=True)[0].item()

    # Get top-k predictions for that position
    probs = torch.softmax(logits[0, mask_index], dim=-1)
    topk = torch.topk(probs, k=5)

    pred_ids = topk.indices.tolist()
    pred_tokens = [tokenizer.decode([i]) for i in pred_ids]

    print("\nTop-5 predictions for the [MASK] token:")
    for i, (tok, score) in enumerate(zip(pred_tokens, topk.values.tolist())):
        print(f"{i+1}. {tok} ({score:.3f})")
