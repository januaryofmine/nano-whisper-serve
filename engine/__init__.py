"""nano-whisper-serve engine — hand-written Whisper-small inference in plain PyTorch.

Modules (built across Milestone 2):
- audio.py     : wav -> log-mel spectrogram (MM 2.1)
- model.py     : Whisper-small architecture, encoder run-once (MM 2.2), decoder (MM 2.3)
- cache.py     : the two KV caches — self-attn (grows) + cross-attn (static) (MM 2.4)
- decode.py    : SOT/suppress startup + greedy loop (MM 2.3)
- tokenizer.py : Whisper BPE (tiktoken) wrapper (MM 2.3)
"""
