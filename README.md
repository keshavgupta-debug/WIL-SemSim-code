# Word Information Lost (WIL) & Semantic Similarity (SemSim) Evaluator

Standalone evaluation toolkit extracted from [Compass Evals](https://github.com/karya-inc/compass-evals) for evaluating Automatic Speech Recognition (ASR) outputs.

---

## 1. Metrics Overview

### Word Information Lost (WIL)
* **Paper:** Morris et al. (2004), *"From WER and RIL to MER and WIL"*.
* **Formula:**
  $$\text{WIL} = 1 - \left(\frac{\text{Hits}}{\text{Ref\_Words}}\right) \times \left(\frac{\text{Hits}}{\text{Hyp\_Words}}\right)$$
  where $\text{Hits} = \text{Ref\_Words} - \text{Substitutions} - \text{Deletions}$.
* **Key Benefit:** Unlike standard WER (which can exceed 100%), WIL is bounded strictly between **0.0 (perfect match)** and **1.0 (total information lost)** while symmetrically penalizing omissions and hallucinations.

### Semantic Similarity (SemSim)
* **Model:** [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE) (Language-agnostic BERT Sentence Embedding), supporting 109+ languages and code-switching dialects.
* **Calculation:** Computes the cosine similarity between L2-normalized sentence embeddings ($\mathbf{u} \cdot \mathbf{v}$).
* **Key Benefit:** Detects whether semantic meaning was preserved when valid synonyms, slang, or paraphrases differ from ground truth.

---

## 2. Installation

```bash
git clone https://github.com/keshavgupta-debug/WIL-SemSim-code.git
cd WIL-SemSim-code

pip install -r requirements.txt
```

---

## 3. Usage

Run the standalone demo script with mock test cases:

```bash
python standalone_wil_and_semsim.py
```

### Programmatic Usage

```python
import pandas as pd
from standalone_wil_and_semsim import evaluate_asr_pairings

data = pd.DataFrame([
    {
        "reference": "i need to purchase groceries",
        "hypothesis": "i have to buy groceries",
    }
])

per_item_results, corpus_summary = evaluate_asr_pairings(data)
print(per_item_results[["reference", "hypothesis", "wer", "wil", "semsim"]])
print(corpus_summary)
```
