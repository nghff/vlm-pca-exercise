# vlm-pca-exercise
An implementation of principle component analysis on popular VLMs including [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [LLaVA-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b), and [MolMo2-8B](https://huggingface.co/allenai/Molmo2-8B). We demonstrate the formation of meaningful object count-based structures in later intermediate layers through reasoning when the VLM is extracting the count of an object.

file/dirs structure:
- `Raphi_Task.ipynb`: notebook responsible for running evaluation on dataset + saving selected activations.
- `Raphi_Task_Results.ipynb`: notebook responible for PCA analysis and things related to discussion.
- `Raphi_Task_Data`: directory containing all the saved results. Subfolders for the three VLMs.
  - `results.csv`: CSV containing results from evaluation (including counts 1-10). One per VLM directory.
  - `activations_data.csv`: CSV containing activations of VLM when dealing with counts 1-5. One per VLM directory.
  - `decoder_fivelayers_pca.png`: PCA plot of the five layers in the decoder, colored by object counts.
  - other PNGs: PCA plots of the activations of various layers. The filenames indicate the layer name and the token id used ('average' if averaged across sequence)

# Brief Model Details
- Architectures
  - All models generally use the same conventional VLM structure of a vision encoder (all three use a ViT-like encoder) that produces visual features/embeddings, which are transformed to language space and used as typical token embeddings in the transformer decoder. Some specialized positional embeddings like MRoPE are applied in the decoder in order to encode spatial information.
  - Interestingly, Qwen is special and has DeepStack Integration, which takes visual tokens from three layers (early, middle, late) in the vision encoder and feeds them through extra transformer layers. The information is incorporated later in the decoder via skip connections to reinforce or further extract helpful visual information.
- Layer counts

<div align="center">
  
  | Model | Vision Encoder Blocks | Language Decoder Blocks | Total Modules (double counting hierarchically) |
  | --- | --- | --- | --- |
  | Qwen3-VL-2B-Instruct | 24 | 28 | 695 |
  | llava-1.5-7b-hf | 24 | 32 | 725 |
  | Molmo2-8B | 25 | 36 | 819 |
  
Table of block counts and total layers. The total modules may double count because it was found using a quick `print(len(list(model.named_modules())))`.

</div>

# Evaluating the Models

## Dataset Quirks

The dataset used for evaluation is [PixMo-Count](https://huggingface.co/datasets/allenai/pixmo-count/viewer/default/test). Note that the original test split is "human-verified and only contain counts from 2 to 10."

## Evaluation Results

<div align="center">

  | Model        | Accuracy | Mean Error | MSE  |
  |--------------|----------|------------|------|
  | Qwen3-VL-2B  | 62.8%    | 0.57       | 1.39 |
  | LLaVA-1.5-7B | 33.0%    | 1.54       | 5.32 |
  | MolMo2-8B    | 74.4%    | 0.37       | 1.02 |

</div>

Table of evaluation results over counts 0-10 (really, 2-10 due to counts in original set).

# PCA Methodology and Observations
For simplicity, we only consider counts 0-5 (really, 2-5) for the PCA analysis.

## Choosing Layer Activations
We initially chose to gather intermediate activations from five layers: first two, middle, late, and final blocks in the decoder. Other activations from decoder attention weights and early, middle, and late vision blocks were gathered. All images can be viewed in the github repo.

The motivation for not PCA-ing on intermediate activations from the vision encoder is that the vision encoder blocks are not being finetuned to extract count features. In particular, since it is not finetuned, the intermediate activations will only reflect a compression of all useful features in the VLM tasks that it was originally trained for. Thus, it is less likely that, when PCA'ed, hidden states produced by vision blocks will exhibit meaningful visual separations between samples of different object count.

On the other hand, even if the decoder is not finetuned on this counting task, intermediate activations may exhibit meaningful separation when PCA'ed. This is because, unlike the visual encoder, the decoder can attend to ALL tokens, including language tokens. Importantly, the language embeddings contain information about the counting task, and the decoder transformer may reason through extracting the relevant 'count features' from the visual embeddings. Thus, we may want to try PCA-ing the intermediate activations from the decoder to see what interesting structures may arise from reasoning on a counting task.

The motivation for picking first two, middle, late, and final layers from the decoder is that the separation may happen gradually or suddenly, and we are curious to see the process from start to end through the entier decoder.

## Choosing Sequence Idx
Note that in a decoder-only transformer, the language-model (LM) head is applied position-wise: each token position $t$ produces a hidden state $h_t$ that is mapped to logits used to predict the **next** token $x_{t+1}$. Thus, during autoregressive generation, the next token is predicted from the hidden state at the **final non-padding position**. Because we use `tokenizer.padding_side = "left"`, padding tokens appear on the left, so the final non-pad token is at the **last sequence index**. Since the model's count prediction is expressed through the next-token logits at this final position, the effects of examples having different object counts will be more noticable at this index. 

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

## Emergence of Clusters by Count
<div align="center">
  
  <img width="100%" height="811" alt="image" src="https://github.com/user-attachments/assets/d9dc8d09-6c6e-48d7-aa63-f610174c2828" />
  <img width="66%" height="811" alt="image" src="https://github.com/user-attachments/assets/dd9aa362-2738-441e-b632-d409bb095d71" />
  
  In reading order: hidden states of blocks 0, 1, 13, 20, and 27, respectively. This is for the Qwen model. (Note that the scale for counts ranges from 2-5 rather than 0-5 because the original test split was missing examples with counts 0 or 1.)
  
</div>

We found that decoder-layer representations of samples with identical object counts progressively form emergent clusters starting around the mid-layers of the decoder, and they become largely separated by the final layer.

## PCA Differences Between Layers

The progression of the PCA plot through the decoder layers is as follows:
1. In early decoder blocks 0 and 1, the PCA plot lacks much structure with colors intermixed, as the model has only just begun to combine visual and language information. There is no outstanding signal for the count, and thus the PCA is not able to separate the colors.
2. In middle/late decoder blocks (13 and 20 for Qwen), the PCA plot begins to show separation of colors, as the model begins to form higher level features, possibly concerning the object counts. There is a stronger signal from the count, and thus more variance is coming from the counts. However, the signal is stil mixed with other variation. The separation is noticeably more obvious in the late block than in the middle block.
3. In the final decoder block (27 for Qwen), the PCA plot shows an effective separation of the colors, consistent with the model's representation at the last token position becoming strongly aligned with the information needed to produce the next token (the count). Thus, a good amount of the variance is explained by the count, and the PCA is able to separate the colors well.

# Functional Differences Between Layers
We will briefly describe the general differences between layers in VLMs. All three use a similar architecture or patterns of layers, so we will describe each component once.

- Vision Encoder: The Vision Encoder does the work of extracting useful visual information in the form of visual embeddings. In all three models, it is a ViT. (Fascinatingly, to perform decently, the vision encoder does not necessarily need to recieve information from the language prompt in order to decide useful information)
  - Vision Block: In all three models, there is a repeating group of layers that is generally a 'vision block'. Part of the ViT, these are transformer blocks with all the usual multi-head attention, except now it operates on patch embeddings rather than token embeddings. This allows image patches to 'attend' to one another like tokens can in regular transformers.
  - Merger: There is a final part of the Vision Encoder that maps the output embeddings to a language space to be used by the decoder.
- VLM Decoder: The VLM Decoder is a transformer that takes token embeddings from all modes (image, text, maybe video) and produces a next token distribution.
  - Block: transformer block with self attention.
  - LM Head: MLP usually for obtaining token distribution with softmax.
