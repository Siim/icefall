import torch
import torchaudio
from xlsr_encoder import XLSREncoder

def test_xlsr_encoder():
    # Initialize encoder with pretrained model
    encoder = XLSREncoder("facebook/wav2vec2-xls-r-300m")
    
    # Test with sample Estonian audio
    waveform, sample_rate = torchaudio.load("test_audio_et.wav")  # 1-2s Estonian speech
    assert sample_rate == 16000, "Resample audio to 16kHz first"
    
    # Forward pass
    with torch.no_grad():
        outputs, lengths = encoder(waveform.unsqueeze(0), torch.tensor([waveform.shape[-1]]))
    
    # Verify output shapes
    assert outputs.shape[-1] == 1024, "XLS-R 300M should output 1024-dim features"
    print(f"Encoder test passed! Output shape: {outputs.shape}")

def test_decoder_projection():
    from train import get_decoder_model
    params = AttributeDict(
        decoder_dim=640,
        joiner_dim=512,
        vocab_size=2500  # Match your BPE model
    )
    decoder = get_decoder_model(params)
    
    # Test embedding layer
    dummy_input = torch.randint(0, params.vocab_size, (10,))  # Random token sequence
    emb = decoder.embedding(dummy_input)
    assert emb.shape == (10, params.decoder_dim), "Decoder embedding mismatch"
    print("Decoder projection test passed!")

if __name__ == "__main__":
    test_xlsr_encoder()
    test_decoder_projection() 