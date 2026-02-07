# vlm-pca-exercise
An implementation of principle component analysis on popular VLMs including [QWEN3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [LLaVA-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b), and [MolMo2](https://huggingface.co/allenai/Molmo2-8B). We demonstrate the formation of meaningful object count-based structures in later intermediate layers through reasoning when the VLM is extracting the count of an object.

# Brief Model Details
- Architectures
  - All models generally use the same conventional VLM structure of a vision encoder (all three use a ViT-like encoder) that produces visual features/embeddings, which are transformed to language space and used as typical token embeddings in the transformer decoder. Some specialized positional embeddings like RoPE are applied in order to encode spatial information.
  - Interestingly, Qwen is special and has a 'deepstack merger' that takes visual tokens from three layers (early, middle, late) in the vision encoder and feeds them through extra transformer layers. The information is incorporated later in the decoder via additive residual connections to reinforce or further extract helpful visual information.
- Layers

<div align="center">
  
  | Model | Vision Encoder Blocks | Language Decoder Blocks | Total Layers (nested double counting) |
  | --- | --- | --- | --- |
  | Qwen3-VL-2B-Instruct | 24 | 28 | TODO |
  | llava-1.5-7b-hf | 24 | 32 | 725 |
  | Molmo2-8B | 25 | 36 | 819 |

</div>

# PCA Methodology and Observations
## Choosing Layer Activations
We initially chose to gather intermediate activations from five layers: first two, middle, late, and final blocks in the decoder. Other activations from decoder attention weights and early, middle, and late vision blocks were gathered. All images can be viewed in the github directory.

The motivation for not PCA-ing on intermediate activations from the vision encoder is that the vision encoder blocks are not being finetuned to extract count features. In particular, since it is not finetuned, the intermediate activations will only reflect a compression of all useful features in the VLM tasks that it was originally trained for. Thus, it is less likely that, when PCA'ed, hidden states produced by vision blocks will exhibit meaningful visual separations between samples of different object count.

On the other hand, even if the decoder is not finetuned on this counting task, intermediate activations may exhibit meaningful separation when PCA'ed. This is because, unlike the visual encoder, the decoder can attend to ALL tokens, including language tokens. Importantly, the language embeddings contain information about the counting task, and the decoder transformer may reason through extracting the relevant 'count features' from the visual embeddings. Thus, we may want to try PCA-ing the intermediate activations from the decoder to see what interesting structures may arise from reasoning on a counting task.

The motivation for picking first two, middle, late, and final layers from the decoder is that the separation may happen gradually or suddenly, and we are curious to see the process from start to end through the entier decoder.

## Choosing Sequence Idx
Note that the classification head of a decoder transformer is tokenwise as a byproduct of independence between tokens due to a need for parallelized causal training, with each token position being used to predict the token in the next position. Thus, the output token embedding most correlated to the next token will be the LAST token index (we used tokenizer.padding_side = 'left' during inference so pad tokens are on the left). 

Because of this, we chose to perform PCA using intermmediate activations at the last sequence index to get the most relevant results.

## Different Layers have different PCA plots
<div align="center">
  
  <img width="900" height="811" alt="image" src="https://github.com/user-attachments/assets/d9dc8d09-6c6e-48d7-aa63-f610174c2828" />
  <img width="600" height="811" alt="image" src="https://github.com/user-attachments/assets/dd9aa362-2738-441e-b632-d409bb095d71" />
  
  In reading order: hidden states of blocks 0, 1, 13, 20, and 27, respectively. This is for the Qwen model.
  
</div>

# Differentiating Layers
We will describe the general functions of different layers in VLMs. All three use a similar architecture or patterns of layers, so we will describe each component once.
- Vision Encoder: The Vision Encoder does the work of extracting useful visual information in the form of visual embeddings. In all three models, it is a ViT. (Fascinatingly, to perform decently, the vision encoder does not necessarily need to recieve information from the language prompt in order to decide useful information)
  - Vision Block: In all three models, there is a repeating group of layers that is generally a 'vision block'. Part of the ViT, these are transformer blocks with all the usual multi-head attention, except now it operates on patch embeddings rather than token embeddings. This allows image patches to 'attend' to one another like tokens can in regular transformers.
  - Merger: There is a final part of the Vision Encoder that maps the output embeddings to a language space to be used by the decoder.
- VLM Decoder: The VLM Decoder is a transformer that takes token embeddings from all modes (image, text, maybe video) and produces a next token distribution.
  - Block: transformer block with self attention.
  - LM Head: MLP usually for obtaining token distribution with softmax.
