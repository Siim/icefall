#!/usr/bin/env python3

import argparse
import asyncio
import grpc
import logging
import numpy as np
import torch
from concurrent import futures
from pathlib import Path
from typing import Iterator

from decode import StreamingEncoder, get_params, get_transducer_model
import sentencepiece as spm
from beam_search import modified_beam_search

# Import generated protobuf code
import asr_service_pb2
import asr_service_pb2_grpc

class ASRServicer(asr_service_pb2_grpc.ASRServiceServicer):
    def __init__(
        self,
        model_path: str,
        bpe_model: str,
        use_xlsr: bool = True,
        xlsr_model_name: str = "facebook/wav2vec2-xls-r-300m",
        chunk_size: int = 8000,  # 0.5s at 16kHz
        beam_size: int = 4
    ):
        """Initialize ASR servicer
        
        Args:
            model_path: Path to trained transducer model
            bpe_model: Path to sentencepiece model
            use_xlsr: Whether to use XLSR encoder
            xlsr_model_name: Name of XLSR model to use
            chunk_size: Size of audio chunks to process
            beam_size: Beam size for decoding
        """
        # Load params
        params = get_params()
        params.use_xlsr = use_xlsr
        params.xlsr_model_name = xlsr_model_name
        
        # Load model
        self.model = get_transducer_model(params)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # Initialize streaming encoder
        self.streaming_encoder = StreamingEncoder(
            encoder=self.model.encoder,
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 2,
            adaptive_chunk_size=True,
            min_chunk_size=4000,
            max_chunk_size=12000,
            latency_tolerance=0.1
        )
        
        # Load BPE model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(bpe_model)
        
        # Decoding params
        self.beam_size = beam_size
        
        logging.info(f"ASR Service initialized on {self.device}")
        
    async def StreamingRecognize(
        self,
        request_iterator: Iterator[asr_service_pb2.AudioRequest],
        context: grpc.aio.ServicerContext
    ) -> Iterator[asr_service_pb2.RecognitionResponse]:
        """Process streaming audio requests
        
        Args:
            request_iterator: Iterator over audio chunks
            context: gRPC context
        
        Yields:
            Recognition responses containing decoded text
        """
        # Reset encoder state
        self.streaming_encoder.reset()
        
        async for request in request_iterator:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            # Convert audio bytes to tensor
            audio = torch.frombuffer(request.audio_content, dtype=torch.float32)
            audio = audio.unsqueeze(0)  # Add batch dimension
            
            if self.device.type == "cuda":
                audio = audio.cuda()
            
            # Get encoder output
            encoder_out, _ = self.streaming_encoder(audio)
            
            # Decode with beam search
            hyp_tokens = modified_beam_search(
                model=self.model,
                encoder_out=encoder_out,
                beam=self.beam_size
            )
            
            # Convert to text
            text = self.sp.decode(hyp_tokens)
            
            end_time.record()
            torch.cuda.synchronize()
            latency = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            
            # Send response
            yield asr_service_pb2.RecognitionResponse(
                text=text,
                is_final=request.is_last,
                confidence=0.9,  # TODO: Implement proper confidence scoring
                latency=latency
            )

def serve(
    model_path: str,
    bpe_model: str,
    port: int = 50051,
    use_xlsr: bool = True,
    xlsr_model_name: str = "facebook/wav2vec2-xls-r-300m"
):
    """Start ASR server
    
    Args:
        model_path: Path to trained model
        bpe_model: Path to sentencepiece model
        port: Port to listen on
        use_xlsr: Whether to use XLSR encoder
        xlsr_model_name: Name of XLSR model to use
    """
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
    asr_service_pb2_grpc.add_ASRServiceServicer_to_server(
        ASRServicer(
            model_path=model_path,
            bpe_model=bpe_model,
            use_xlsr=use_xlsr,
            xlsr_model_name=xlsr_model_name
        ),
        server
    )
    
    server.add_insecure_port(f'[::]:{port}')
    return server

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained transducer model"
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        required=True,
        help="Path to sentencepiece model"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to listen on"
    )
    parser.add_argument(
        "--use-xlsr",
        type=bool,
        default=True,
        help="Whether to use XLSR encoder"
    )
    parser.add_argument(
        "--xlsr-model-name",
        type=str,
        default="facebook/wav2vec2-xls-r-300m",
        help="Name of XLSR model to use"
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        level=logging.INFO
    )
    
    server = await serve(
        model_path=args.model_path,
        bpe_model=args.bpe_model,
        port=args.port,
        use_xlsr=args.use_xlsr,
        xlsr_model_name=args.xlsr_model_name
    )
    
    logging.info(f"Server starting on port {args.port}")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(main()) 