# vlm-pca-exercise
An implementation of principle component analysis on popular VLMs including [QWEN3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [LLaVA-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b), and [MolMo2](https://huggingface.co/allenai/Molmo2-8B). We demonstrate the formation of meaningful object count-based structures in later intermediate layers through reasoning when the VLM is extracting the count of an object.

# Brief Model Details
- Architectures
  - All models generally use the same conventional VLM structure of a vision encoder (all three use a ViT) that produces visual features/embeddings, which are transformed to language space and used as typical token embeddings in the transformer decoder. Some specialized positional embeddings like RoPE are applied in order to encode spatial information.
  - Interestingly, Qwen is a special with a 'deepstack merger' that takes visual tokens from three layers (early, middle, late) in the vision encoder and feeds them through extra transformer layers. The information is incorporated later in the decoder via additive residual connections to reinforce helpful visual information.
- Layers

<div align="center">
  
| Model | Vision Encoder Blocks | Language Decoder Blocks | Total Layers (nested double counting) |
| --- | --- | --- | --- |
| Qwen3-VL-2B-Instruct | 24 | 28 | TODO |
| llava-1.5-7b-hf | 24 | 32 | 725 |
| Molmo2-8B | 25 | 36 | 819 |

</div>

- Describing Layers
  - Vision Encoder: The Vision Encoder does the work of extracting useful visual information in the form of visual embeddings. (Fascinatingly, to perform decendtly, the vision encoder does not necessarily need to recieve information from the language prompt in order to decide useful information)
  - VLM Decoder: 
