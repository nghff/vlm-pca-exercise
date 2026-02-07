# vlm-pca-exercise
An implementation of principle component analysis on popular VLMs including [QWEN3-VL](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [LLaVA-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-7b), and [MolMo2](https://huggingface.co/allenai/Molmo2-8B). We demonstrate the formation of meaningful object count-based structures in later intermediate layers through reasoning when the VLM is extracting the count of an object.

# Brief Model Details
- Architectures
  - All models generally use the same conventional VLM structure of a vision encoder (all three use a ViT-like encoder) that produces visual features/embeddings, which are transformed to language space and used as typical token embeddings in the transformer decoder. Some specialized positional embeddings like RoPE are applied in order to encode spatial information.
  - Interestingly, Qwen is special and has a 'deepstack merger' that takes visual tokens from three layers (early, middle, late) in the vision encoder and feeds them through extra transformer layers. The information is incorporated later in the decoder via skip connections to reinforce or further extract helpful visual information.
- Layer counts

<div align="center">
  
  | Model | Vision Encoder Blocks | Language Decoder Blocks | Total Modules (double counting hierarchically) |
  | --- | --- | --- | --- |
  | Qwen3-VL-2B-Instruct | 24 | 28 | 695 |
  | llava-1.5-7b-hf | 24 | 32 | 725 |
  | Molmo2-8B | 25 | 36 | 819 |
  
Table of block counts and total layers. The total modules may double count because it was found using a quick `print(len(list(model.named_modules())))`.

</div>

# PCA Methodology and Observations
## Choosing Layer Activations
We initially chose to gather intermediate activations from five layers: first two, middle, late, and final blocks in the decoder. Other activations from decoder attention weights and early, middle, and late vision blocks were gathered. All images can be viewed in the github repo.

The motivation for not PCA-ing on intermediate activations from the vision encoder is that the vision encoder blocks are not being finetuned to extract count features. In particular, since it is not finetuned, the intermediate activations will only reflect a compression of all useful features in the VLM tasks that it was originally trained for. Thus, it is less likely that, when PCA'ed, hidden states produced by vision blocks will exhibit meaningful visual separations between samples of different object count.

On the other hand, even if the decoder is not finetuned on this counting task, intermediate activations may exhibit meaningful separation when PCA'ed. This is because, unlike the visual encoder, the decoder can attend to ALL tokens, including language tokens. Importantly, the language embeddings contain information about the counting task, and the decoder transformer may reason through extracting the relevant 'count features' from the visual embeddings. Thus, we may want to try PCA-ing the intermediate activations from the decoder to see what interesting structures may arise from reasoning on a counting task.

The motivation for picking first two, middle, late, and final layers from the decoder is that the separation may happen gradually or suddenly, and we are curious to see the process from start to end through the entier decoder.

## Choosing Sequence Idx
Note that in a decoder-only transformer, the language-model (LM) head is applied **position-wise**: each token position $t$ produces a hidden state $h_t$ that is mapped to logits used to predict the **next** token $x_{t+1}$. While training is parallelized across positions, tokens are not independent—each position attends to previous tokens under a causal mask. During autoregressive generation, the next token is predicted from the hidden state at the **final non-padding position**. Because we use `tokenizer.padding_side = "left"`, padding tokens appear on the left, so the final non-pad token is at the **last sequence index**.

Since the model's count prediction is expressed through the next-token logits at this final position, the effects of examples having different object counts will be more noticable at this index. Therefore, 

Because of this, we chose to perform PCA on intermediate activations at the last sequence index to maximize the chance that the projection separates samples along count-related variation.

## Alternative Sequence Ids
The PCA plots for these can also be found in this github repo.
1. Object token index - we could also try performing PCA at the index with the token representing the object to count (e.g. if the task is to count the number of "planes," then we do PCA at the index with the token for "planes").
  a. Pros: More directly targets the representation tied to the queried object, which can make clusters reflect specific object-related features rather than generic answer-generation dynamics. It could also be interesting to see how the different object tokens contribute to clustering in PCA plots.
  b. Cons: The index of the object token in the prompt is delicate (this is not really a concern in this case with a fixed prompt structure), and some object words may be split into more than one object tokens.

3. Average over all sequence ids - this would involve taking an average of hidden state embeddings over all sequence indicies.
  a. Pros: The resulting average is more stable and less prone to outliers and noise. It can be interesting to see the global patterns in how the VLM predicts every token in the seqeuence.
  b. Cons: Any one signal from a single position will be washed out and mixed with others. The result can be less useful for what we do when the most important information for the object counting is likely to be at the last position.

## Dimensions

<div align="center">
  
  | Model | Pre-PCA | Post_PCA |
  | --- | --- | --- |
  | Qwen3-VL-2B-Instruct | 2048 | 2 |
  | llava-1.5-7b-hf | 4096 | 2 |
  | Molmo2-8B | 4096 | 2 |
  
Table of single token position activation dimension sizes before and after PCA projection.

</div>

The dimensions of the activations at the decoder layers are $(T, D)$, where $T$ is the number tokens in a sequence, and $D$ is the internal dimension of the transformer token embeddings. When fed into PCA, the activation at a single token index is chosen, and thus it will be shape $(D)$. 

## Different Layers have different PCA plots
<div align="center">
  
  <img width="100%" height="811" alt="image" src="https://github.com/user-attachments/assets/d9dc8d09-6c6e-48d7-aa63-f610174c2828" />
  <img width="66%" height="811" alt="image" src="https://github.com/user-attachments/assets/dd9aa362-2738-441e-b632-d409bb095d71" />
  
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
