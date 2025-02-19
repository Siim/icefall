#!/usr/bin/env python3

import argparse
import asyncio
import grpc
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import AsyncIterator

import asr_service_pb2
import asr_service_pb2_grpc

async def stream_audio(
    audio_path: str,
    chunk_size: int = 8000  # 0.5s at 16kHz
) -> AsyncIterator[asr_service_pb2.AudioRequest]:
    """Stream audio file in chunks
    
    Args:
        audio_path: Path to audio file
        chunk_size: Size of audio chunks in samples
        
    Yields:
        AudioRequest objects containing audio chunks
    """
    # Load audio file
    audio, sample_rate = sf.read(audio_path)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sample_rate}Hz")
    
    # Convert to float32 if needed
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Stream in chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        
        # Zero pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        yield asr_service_pb2.AudioRequest(
            audio_content=chunk.tobytes(),
            is_last=(i + chunk_size >= len(audio))
        )
        
        # Small delay to simulate real-time streaming
        await asyncio.sleep(0.1)

async def transcribe_file(
    stub: asr_service_pb2_grpc.ASRServiceStub,
    audio_path: str,
    chunk_size: int = 8000
):
    """Transcribe audio file using streaming ASR
    
    Args:
        stub: gRPC stub
        audio_path: Path to audio file
        chunk_size: Size of audio chunks in samples
    """
    print(f"Transcribing {audio_path}")
    
    # Stream audio and get responses
    audio_stream = stream_audio(audio_path, chunk_size)
    async for response in stub.StreamingRecognize(audio_stream):
        print(f"Transcript: {response.text}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Latency: {response.latency*1000:.1f}ms")
        print(f"Final: {response.is_final}")
        print("---")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server",
        type=str,
        default="localhost:50051",
        help="Address of ASR server"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8000,
        help="Size of audio chunks in samples"
    )
    args = parser.parse_args()
    
    # Create channel and stub
    async with grpc.aio.insecure_channel(args.server) as channel:
        stub = asr_service_pb2_grpc.ASRServiceStub(channel)
        
        # Transcribe file
        await transcribe_file(
            stub=stub,
            audio_path=args.audio_path,
            chunk_size=args.chunk_size
        )

if __name__ == "__main__":
    asyncio.run(main()) 