# Quantifying Creativity in LLMs

`#creativity`, `#evaluation`, `#metric`, `#LLMs`

Abeen Bhattacharya

# Abstract

The quantification of creativity in large language models (LLMs) is essential given their increasing influence across domains such as poetry, technical writing, and storytelling. In this project, we introduce a novel Creativity Index (CI) that integrates measures of statistical novelty, semantic coherence, and contextual appropriateness to assess the creativity of LLM‐generated text. In addition, we develop four neural models that combine these metrics with raw textual input and train them on a dataset of 2,002 contrastive pairs (1,063 literary and 939 technical) judged by GPT‐4. This evaluation framework examines whether the metrics we use in constructing the CI align with LLMs’ own conceptualization of creativity.

---

# What This Project Is About

Large language models (LLMs) generate outputs ranging from sonnets to technical abstracts and narratives steeped in magical realism. Yet, a standardized metric for assessing creativity has been lacking. We address this gap by pursuing two primary objectives:
1. **Developing a Quantitative Metric for Creativity:**  
   We create a composite Creativity Index (CI) by integrating multiple metrics (novelty, coherence, contextual fit, syntactic complexity, lexical diversity, lexical novelty, surprise, emotional expressiveness, and human-likeness).
2. **Understanding Creativity in LLMs:**  
   We investigate whether these metrics correspond to the way LLMs “perceive” creativity by training neural models on contrastive pairs (judged by GPT-4) and comparing their scores with our fixed CI.

_Text is segmented using SpaCy, and our models are trained contrastively using margin ranking loss._


---

# Progress Made So Far
- **Dataset Creation:**  
  - **Contrastive Pairs:** A total of **2,002** contrastive pairs were created: **1,063** pairs from the literary domain and **939** from the technical domain.  
  - **Dataset Creation Process:**  
    - **Prompt Generation:** Using a GPT-2–based prompt generator, we sampled 10 creative prompts per domain from baseline corpora (Project Gutenberg for literary and arXiv abstracts for technical).
    - **Text Generation:** For each prompt, two texts were generated using GPT-4/OpenAI.
    - **Contrastive Judgment:** GPT-4 was then used to determine which text was “more creative.”
    - **Metric Computation:** All nine creativity metrics were computed for each segment of both texts.
    - **Data Storage:** The contrastive pairs (text segments and their metric vectors) were stored for training.

- **Model Training:**  
  Four neural models were developed:
  1. **CompositeRegressor:** Combines DistilBERT embeddings with an LSTM on metric features.
  2. **SimpleCreativityPredictor:** Uses DistilBERT embeddings processed by an LSTM (text-only).
  3. **TransformerCreativityAggregator:** Uses a transformer encoder to combine metric interactions with text embeddings.
  4. **TextBasedCreativityPredictor:** Uses DistilBERT to generate weights that re-calculate the CI formula.

  Training used a contrastive learning approach with a Margin Ranking Loss. 

- **Evaluation:**  
  Automated testing was conducted using fixed test prompts per domain. Generated texts were evaluated with our Fixed CI as well as the scores from each neural model. Additional graphical analyses (scatter plots, histograms, and heatmaps) were produced.


## Approach

### Main Approach

Creativity is quantified using nine normalized metrics:

1.  **Novelty (N)**\
    KL divergence between token distributions:

    `D_KL(P || Q) = Σ P(t) * log( P(t) / Q(t) )`

2.  **Coherence (C)**\
    Inverse GPT-2 perplexity:

    `C = 1 / ( e^(loss) + 10^-6 )`

3.  **Contextual Fit (CT)**\
    Sentence-BERT cosine similarity:

    `CT = (A - B) / (||A|| * ||B||)`

4.  **Syntactic Complexity (SC)**\
    Mean of scaled sub-metrics (e.g., sentence length, clause count, parse tree depth, POS entropy).

5.  **Lexical Diversity (LD)**\
    Scaled weighted mean of n-gram diversity and inverted Self-BLEU.

6.  **Lexical Novelty (LN)**\
    Proportion of unique n-grams (n ≥ 5) not in a baseline corpus, computed via DJ Search.

7.  **Surprise (S)**\
    Mean cosine distance between consecutive sentence embeddings.

8.  **Emotional Expressiveness (EE)**\
    Variance of VADER sentiment scores per sentence.

9.  **Human-Likeness (HL)**\
    Exponentiated GPT-2 perplexity or a MAUVE score capturing overall text plausibility.

* * * * *

These metrics are then aggregated in two ways:

1.  **Fixed CI**\
    A simple weighted sum of normalized metrics:

    `CI = Σ [w_i * norm(m_i)]    where Σ w_i = 1   (default w_i = 0.111)`

2.  **Neural Models**

    -   **CompositeRegressor**\
        Uses DistilBERT + LSTM + MLP for a hybrid text-and-metrics input.
    -   **SimpleCreativityPredictor**\
        A text-only baseline with DistilBERT + LSTM.
    -   **TransformerCreativityAggregator**\
        DistilBERT + a small Transformer (4 layers, 4 heads) to learn interactions among the metrics.
    -   **TextBasedCreativityPredictor**\
        DistilBERT + MLP that dynamically learns metric weights.

Text is segmented with SpaCy, and the four neural models are trained contrastively using margin ranking loss on GPT-4○ judgments.

### Baselines

The fixed CI (equal weights) serves as the baseline, compared against neural model scores.

### Dataset Creation

The training dataset of 2,002 contrastive pairs was created as follows:

1.  **Prompt Generation**: 10 imaginative prompts per domain (literary: e.g., "Quantum sonnet"; technical: e.g., "Encryption abstract") were crafted using GPT-2 and random sampling from baseline corpora. One prompt was selected per iteration.
2.  **Response Generation**: For each prompt, GPT-4o generated two responses.
3.  **Judgment**: GPT-4o judged which response was more creative based on its internal understanding.
4.  **Metric Computation**: All nine metrics were computed for each response against Project Gutenberg (literary) or arXiv abstracts (technical).
5.  **Storage**: Judgments, text pairs, and metrics were stored in .pkl files (1,063 literary, 939 technical pairs loaded from existing files).

# Experiments

### Data

-   **Datasets**:
    -   **Project Gutenberg**: ~5.5M words of literary texts for poetic/narrative creativity.
    -   **arXiv Abstracts**: ~1M words of technical papers for scientific innovation.
-   **Task**: Assess creativity in GPT-4 and Together AI outputs across literary and technical prompts.

### Evaluation Method

-   **Metrics**: Fixed CI, neural model scores, Pearson/Spearman correlations between model predictions and CI/GPT-4o judgments.
-   **Analysis**: Compared fixed CI to model scores and evaluated Together AI model-specific strengths.

### Experimental Details

-   **Setup**: Python 3.11, Google Colab, PyTorch 2.x, Tesla T4 GPU (16 GB).
-   **Hyperparameters**:
    -   **CompositeRegressor**: LSTM (hidden=64) + MLP, lr=1e-3, batch size=8, AdamW (weight decay=1e-5), 50 epochs.
    -   **SimpleCreativityPredictor**: LSTM (hidden=64) + MLP, lr=1e-3, batch size=8, AdamW, 50 epochs.
    -   **TransformerCreativityAggregator**: Transformer (4 layers, 4 heads, hidden=64), lr=1e-3, batch size=8, AdamW, 50 epochs.
    -   **TextBasedCreativityPredictor**: MLP (3 layers, hidden=64), lr=2e-5, batch size=8, AdamW, 100 epochs.
-   **Generation**: Temperature=0.8, top-p=0.95, max_tokens=200.
-   **Pairs**: 2,002 pairs (1,063 literary, 939 technical), split 80% train, 20% validation.

## Results

## Training Loss Trends

When training with only 500 pairs the models achieved significantly lower (i.e. “better”) best validation losses in fewer epochs, but when we increased the dataset size the best validation losses became worse and the convergence was more challenging. This suggests that even LLM‐based evaluators may not be completely objective when judging creativity across larger, more diverse contrastive pair sets.

**Trends (2002 Pairs)**
-------------------------------------

| **Model** | **Best Val Loss** | **Epoch** | **Train Loss** |
| --- | --- | --- | --- |
| **SimpleCreativityPredictor** | 0.8390 | 19 | 0.7725 |
| **CompositeRegressor** | 0.8332 | 19 | 0.7725 |
| **TransformerCreativityAggregator** | 0.7906 | 28 | 0.6571 |
| **TextBasedCreativityPredictor** | 0.8340 | 14 | 0.7362 |

> **Note:**
>
> -   *SimpleCreativityPredictor* and *TextBasedCreativityPredictor* each eventually stopped early around those epochs, with the listed Val Loss being the best they achieved (even though some logs showed epochs continuing, the final "best" was reached by the epoch noted).
> -   The *CompositeRegressor* and *SimpleCreativityPredictor* both happened to converge best at epoch 19 (though on different runs/logs), showing the same final training loss (0.7725) but slightly different best Val Losses (0.8332 vs. 0.8390).
> -   *TransformerCreativityAggregator* improved more gradually up through epoch 28, achieving the lowest Val Loss of the four on this larger (2002) dataset.
> -   The above training trends was taken such that the least validation loss was reached, across 5 seeds of training

**Trends (500 Pairs)**
------------------------------------

| **Model** | **Best Val Loss** | **Epoch** | **Train Loss** |
| --- | --- | --- | --- |
| **SimpleCreativityPredictor** | 0.7423 | 17 | 0.7222 |
| **CompositeRegressor** | 0.6259 | 14 | 0.6174 |
| **TransformerCreativityAggregator** | 0.5835 | 24 | 0.5956 |
| **TextBasedCreativityPredictor** | 0.6183 | 39 | 0.6139 |

---

### Testing Results

#### Literary Domain


![literary_CompositeRegressor_scatter](https://github.com/user-attachments/assets/45b71df6-c7af-4821-8549-806a0dd0439d)
![literary_SimpleCreativityPredictor_scatter](https://github.com/user-attachments/assets/fbb959fc-4b7b-45bd-9cd6-16a5e4f232d7)
![literary_TextBasedCreativityPredictor_scatter](https://github.com/user-attachments/assets/375b098e-1866-4776-912a-82a221169d0e)
![literary_TransformerCreativityAggregator_scatter](https://github.com/user-attachments/assets/a342edb8-2d90-4263-94d4-61af2b36b7f7)

The table below presents the top performances of different model types in the literary domain, focusing on prompts such as "Quantum Sonnet," "Short Story," and "Magical Realism." Each row reflects the best score achieved by a specific model type for a given prompt, offering a clear view of their effectiveness in literary creativity.

| **Prompt**             | **Fixed CI** | **Best Model Score** | **Model Type**                     |
|------------------------|--------------|----------------------|------------------------------------|
| **Quantum Sonnet**     | 0.5152       | 9.92                 | CompositeRegressor                 |
|                        | 0.5103       | 9.67                 | TransformerCreativityAggregator    |
|                        | 0.4596       | 5.22                 | SimpleCreativityPredictor          |
|                        | 0.4269       | 0.71                 | TextBasedCreativityPredictor       |
| **Short Story**        | 0.4345       | 5.33                 | CompositeRegressor                 |
|                        | 0.4164       | 4.16                 | TransformerCreativityAggregator    |
|                        | 0.3808       | 0.99                 | SimpleCreativityPredictor          |
|                        | 0.3299       | 0.40                 | TextBasedCreativityPredictor       |
| **Magical Realism**    | 0.4203       | 6.65                 | CompositeRegressor                 |
|                        | 0.4087       | 6.10                 | TransformerCreativityAggregator    |
|                        | 0.3927       | 4.15                 | SimpleCreativityPredictor          |
|                        | 0.3705       | 0.50                 | TextBasedCreativityPredictor       |

#### Observations (Literary):
- **CompositeRegressor**:  
  Achieved the highest score for “Quantum Sonnet” (9.92) and delivered strong performance across all prompts.
- **TransformerCreativityAggregator**:  
  Performed competitively with high scores, especially on imaginative prompts like “Magical Realism.”
- **SimpleCreativityPredictor** and **TextBasedCreativityPredictor**:  
  Produced lower and less consistent scores, indicating their limitations in capturing the full spectrum of literary creativity.

---

#### Technical Domain

![technical_CompositeRegressor_scatter](https://github.com/user-attachments/assets/fb3383a3-3fb0-4618-95b8-eda0f0492466)
![technical_SimpleCreativityPredictor_scatter](https://github.com/user-attachments/assets/414abd73-46d3-498d-9e7d-291fccb32a67)
![technical_TextBasedCreativityPredictor_scatter](https://github.com/user-attachments/assets/145cebd3-d360-426f-97a8-2745002717cf)
![technical_TransformerCreativityAggregator_scatter](https://github.com/user-attachments/assets/a491ed81-d428-43a8-81ea-b58eacd43159)


The table below highlights the top performances of each model type in the technical domain, focusing on prompts like "AI Abstract," "Encryption Abstract," and "Cloud Architecture." Each row shows the best score for a specific model type per prompt, providing insights into their technical creativity strengths.

| **Prompt**             | **Fixed CI** | **Best Model Score** | **Model Type**                     |
|------------------------|--------------|----------------------|------------------------------------|
| **AI Abstract**        | 0.4574       | 6.40                 | TransformerCreativityAggregator    |
|                        | 0.3483       | 5.31                 | CompositeRegressor                 |
|                        | 0.3465       | 2.99                 | SimpleCreativityPredictor          |
|                        | 0.2988       | 0.49                 | TextBasedCreativityPredictor       |
| **Encryption Abstract**| 0.4917       | 8.33                 | TransformerCreativityAggregator    |
|                        | 0.4761       | 7.43                 | CompositeRegressor                 |
|                        | 0.4099       | 5.17                 | SimpleCreativityPredictor          |
|                        | 0.3678       | 0.58                 | TextBasedCreativityPredictor       |
| **Cloud Architecture** | 0.4538       | 6.40                 | TransformerCreativityAggregator    |
|                        | 0.4328       | 5.12                 | CompositeRegressor                 |
|                        | 0.3601       | 0.96                 | SimpleCreativityPredictor          |
|                        | 0.3216       | 0.50                 | TextBasedCreativityPredictor       |

#### Observations (Technical):
- **TransformerCreativityAggregator**:  
  Consistently delivered high scores, reflecting its ability to capture nuanced technical details.
- **CompositeRegressor**:  
  Showed balanced performance, though slightly behind the Transformer model on innovative tasks.
- **SimpleCreativityPredictor** and **TextBasedCreativityPredictor**:  
  Performed at a lower level, highlighting challenges in assessing technical creativity.

## Together AI Model Comparison

We tested multiple prompts spanning both **literary** and **technical** domains against several Together AI models---namely:

-   **meta-llama/Llama-3.3-70B-Instruct-Turbo-Free**
-   **mistralai/Mixtral-8x7B-Instruct-v0.1**
-   **Qwen/Qwen2-72B-Instruct**

and measured each output's **Fixed CI** (our fixed creativity index) alongside the scores assigned by our trained creativity models (e.g., **CompositeRegressor**, **SimpleCreativityPredictor**, **TransformerCreativityAggregator**, **TextBasedCreativityPredictor**).

Below is a broad synthesis of the observed trends:

1.  **Literary Prompts (e.g., "Write a sonnet about quantum physics" or "Generate a creative narrative in the style of magical realism"):**

    -   **Fixed CI** values typically ranged between **0.45--0.58** across the different Together AI models.
    -   **CompositeRegressor** scores, which often spiked for well-structured, thematically rich texts, hovered around **5.0--6.7** in some cases.
    -   **Qwen** models often produced slightly higher CompositeRegressor scores on sonnets (e.g., **~6.47** or more), whereas **mistralai** sometimes had a higher *fixed* CI (e.g., **0.54--0.57** range).
    -   On average, we see that the **TransformerCreativityAggregator** and **CompositeRegressor** give higher numeric scores for outputs that exhibit strong coherence, thematic adherence to the prompt (contextual fit), and lexical diversity (important for literary tasks).
2.  **Technical Prompts (e.g., "Generate a technical description of a cloud computing architecture," "Write a research paper abstract on a new AI algorithm"):**

    -   **Fixed CI** values here commonly landed in the **0.47--0.61** ballpark.
    -   Interestingly, certain outputs with well-structured, precise technical language (e.g., from Qwen or Llama) scored relatively high on the fixed scale but had more moderate CompositeRegressor scores (in the **2--3.5** region). This is because the text is less "creative" in a literary sense yet still satisfies many of the metrics (like coherence and domain fit).
    -   For some prompts, **Qwen** or **mistralai** gave expansions with quite thorough details on cloud computing layers or novel AI methods, achieving above-average **Fixed CI** (above **0.50** in many runs) and moderate to high model scores, especially if the text included novelty in describing the architecture or a distinctive approach to encryption.
3.  **Individual Max Scores & Overall Trends:**

    -   **Highest CompositeRegressor** spikes in literary tasks often appeared in Qwen or mistralai outputs that used vivid, imaginative language (particularly in the "sonnet" or "magical realism" prompts).
    -   For technical tasks, we saw certain Llama or mistralai completions occasionally hitting the upper range of fixed CI---**~0.58--0.61**---particularly if they provided a well-organized breakdown of architecture with novel-sounding features or hybrid solutions.
    -   On average, **mistralai** sometimes gave the highest *fixed* creativity index in a single run (e.g., ~0.57 for a quantum sonnet) but Qwen's text might yield the top CompositeRegressor or TransformerCreativityAggregator scores, signifying more synergy with our internal metrics that reward lexical novelty or structured creativity.

Overall, there is no single "dominant" Together AI model across *all* prompts; each model has pockets of strengths depending on domain and the specific synergy with our creativity metrics. Some produce more "flair" for literary tasks, while others provide more thorough technical detail---boosting the fixed index for domain fit or novelty.
![together_ai_comparison_fixed](https://github.com/user-attachments/assets/8ab63735-94a0-4fa5-918c-7452a0499662)
![together_ai_comparison_CompositeRegressor](https://github.com/user-attachments/assets/249cadfd-0b0e-49c0-b439-79d0f07d437f)
![together_ai_comparison_SimpleCreativityPredictor](https://github.com/user-attachments/assets/5012c84d-89d4-48eb-a332-95ab793202b7)
![together_ai_comparison_TextBasedCreativityPredictor](https://github.com/user-attachments/assets/1db6d28c-e594-46b0-a4d5-828ee5700d26)
![together_ai_comparison_TransformerCreativityAggregator](https://github.com/user-attachments/assets/cb54562a-1c0a-4012-a426-33ad8f20d0df)



## Ablation Study Analysis

We performed **ablation studies** on the nine creativity metrics---**Novelty (N)**, **Coherence (C)**, **Contextual Fit (CT)**, **Syntactic Complexity (SC)**, **Lexical Diversity (LD)**, **Lexical Novelty (LN)**, **Surprise (S)**, **Emotional Expressiveness (EE)**, and **Human-Likeness (HL)**---by omitting one metric at a time and recalculating a normalized composite score. Below are the key takeaways from the ablation plots for each domain:

1.  **Technical Domain Ablation**

![technical_ablation](https://github.com/user-attachments/assets/ad5d1f7a-e15f-4bb4-89ff-ce6f2b6097fb)

In the technical domain, the ablation plot reveals the following:

-   **Syntactic Complexity (SC)** is the most critical metric. Removing it causes the steepest drop in the composite score, falling to approximately **0.45**. This suggests that well-structured and syntactically varied text is essential for perceived creativity in technical writing.
-   **Lexical Diversity (LD)** is also significant, with its omission leading to a composite score of about **0.48**. This indicates that a varied vocabulary plays an important role, though it is not as dominant as syntactic complexity.
-   Other metrics, including **Novelty (N)**, **Coherence (C)**, **Lexical Novelty (LN)**, **Surprise (S)**, **Emotional Expressiveness (EE)**, and **Human-Likeness (HL)**, show a milder impact, with scores remaining closer to **0.5** when omitted. This implies that while these factors contribute to creativity, they are less influential in the technical domain compared to structural and lexical elements.

**Key Insight**: For technical prompts, creativity hinges primarily on the complexity of sentence structures (SC) and the variety of vocabulary (LD). Metrics like emotional expressiveness or surprise, while relevant, are secondary in this context.


2.  **Literary Domain Ablation**

![literary_ablation](https://github.com/user-attachments/assets/5e6fb5f3-f9e1-4ab8-ab73-f4a74b956f93)

In the literary domain, the ablation plot highlights the following:

-   **Novelty (N)** emerges as the most influential metric, with its removal causing the largest drop in the composite score to just above **0.45**. This underscores that originality---fresh imagery and unique phrasing---is paramount to literary creativity.
-   **Coherence (C)** and **Contextual Fit (CT)** are also key contributors, with scores dropping to around **0.47** when either is omitted. This indicates that maintaining a coherent narrative and ensuring contextual appropriateness are vital, though they are slightly less critical than novelty.
-   Metrics such as **Surprise (S)**, **Emotional Expressiveness (EE)**, and **Human-Likeness (HL)** have a more moderate effect, with scores staying closer to **0.5**. This suggests that while unpredictability and emotional depth enhance creativity, they are not as decisive as novelty, coherence, and contextual fit in literary writing.

**Key Insight**: In the literary domain, creativity is driven most strongly by originality (N), followed by narrative coherence (C) and contextual appropriateness (CT). The impact of other metrics is more evenly distributed compared to the technical domain, but none dominate as much as novelty.




---

# Conclusion

We collected a total of **2002** contrastive text pairs---**1063** from the literary domain and **939** from the technical domain---to train and validate our creativity evaluation models. Each pair was generated and then judged by GPT-4 on which text was "more creative," yielding a rich dataset that captures nuanced differences in style, coherence, and originality.

Evaluating various LLM outputs (GPT-4, Together AI's Llama, mistralai, Qwen, etc.) with our **Fixed CI** and four neural models shows that:

-   Different LLMs excel at different styles: Some produce text with high novelty or emotional expressiveness (strong for literary tasks), while others excel at structured clarity or advanced domain fit (better for technical tasks).
-   Our ablation studies highlight the distinct "must-have" metrics for each domain:
    -   In the **technical domain**, **Syntactic Complexity (SC)** is vital, with its omission causing the largest drop in the composite score (to ~0.45), followed by **Lexical Diversity (LD)** (~0.48). This underscores the importance of well-structured, varied sentence construction and a diverse vocabulary for technical creativity.
    -   In the **literary domain**, **Novelty (N)** remains paramount, with its removal leading to the largest drop in the composite score (to just above 0.45). **Coherence (C)** and **Contextual Fit (CT)** are also critical, with scores dropping to ~0.47 when either is omitted, highlighting the need for originality alongside a coherent and contextually appropriate narrative.

By examining these results, we see that creativity in LLMs is multifaceted---reliant on domain-specific demands (structural clarity and lexical variety in technical tasks vs. imaginative flair and narrative coherence in literary tasks) and the synergy of multiple metrics (novelty, syntactic complexity, coherence, contextual fit, etc.). This framework and dataset offer a foundation for future explorations in modeling and enhancing creative text generation across diverse domains.
