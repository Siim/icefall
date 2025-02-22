#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
import torch.nn as nn


class EncoderInterface(nn.Module):
    """Interface for encoders used in transducer models"""
    
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 0  # Must be set by implementing class
        
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, time) or (batch, time, 1)
            x_lens: Length of each sequence in the batch
        Returns:
            (output, output_lens)
            output: (batch, time', output_dim)
            output_lens: (batch,)
        """
        raise NotImplementedError
        
    def streaming_forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Streaming forward pass
        Args:
            x: Input tensor (batch=1, time)
            cache: Optional cached state from previous chunk
        Returns:
            (output, new_cache)
        """
        raise NotImplementedError
        
    def reset_streaming_state(self) -> None:
        """Reset any cached state for streaming inference"""
        pass
