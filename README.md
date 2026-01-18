# CNN Text Classifier — Hyperparameter Sensitivity Experiments

## Overview

This project explores **how sensitive a classical CNN-based text classifier is to common hyperparameter changes**. This is an experimental project rather than a production-ready system.

Unlike my previous transformer-focused project, which emphasized architectural decisions and pipeline optimization, this work intentionally uses a **simple and well-known CNN architecture** and shifts attention to **empirical experimentation**.

The goal was not to improve state-of-the-art performance, but to **observe and analyze training behavior under controlled hyperparameter variations**. 

---

## Motivation

Hyperparameter tuning is often treated as a reliable way to improve model quality.
In practice, however, it is not always clear whether observed metric changes are:

* caused by the hyperparameter itself
* or simply a result of training noise and stochasticity

This project was designed to **test that assumption explicitly**.

> The expectation was that changing key hyperparameters would noticeably affect convergence dynamics or final evaluation metrics — an expectation that was not confirmed by the results.

---

## Experimental Focus

This repository does **not** focus on:

* proposing a novel CNN architecture
* optimizing for maximum benchmark scores
* comparing CNNs to transformers

Instead, it focuses on:

* controlled experimentation
* metric stability
* reproducibility of observed effects

---

## Experimental Setup

All experiments share the same training pipeline:

* IMDB sentiment classification dataset
* fixed train / validation split
* identical optimizer and number of epochs
* identical evaluation protocol

Only **one hyperparameter was modified at a time** relative to the baseline.

---

## Model Variants

| Variant         | Change Relative to Baseline                                 |
| --------------- | ------------------------------------------------------------|
| Default         | Baseline configuration (batch=512, emb_dim=128, n_filt=100) |
| Large Batch     | Batch size increased ×2 (1024)                              |
| Large Embedding | Embedding dimension increased ×2 (256)                      |
| More Filters    | Number of convolution filters increased ×2 (200)            |

---

## Metrics Tracked

During training, the following metrics were logged for validation sets:

* Loss
* Accuracy
* Precision
* Recall
* F1-score

Metrics were recorded per epoch to allow analysis of:

* convergence behavior
* overfitting tendencies
* metric variance

---

## Results Summary

Across all experiments:

* no model variant demonstrated a **stable or qualitatively significant improvement** over the baseline
* differences in F1-score remained within a narrow range
* loss curves exhibited similar convergence patterns
* observed metric fluctuations could not be confidently attributed to specific hyperparameter changes

At the scale of these experiments, **the CNN model appeared largely insensitive to the tested hyperparameters**.

---

## Discussion

This outcome suggests several non-mutually exclusive explanations:

* the baseline configuration may already be sufficient for this dataset
* IMDB sentiment classification may not be sensitive to increased model capacity
* CNN-based text models may reach performance plateaus quickly
* commonly tuned hyperparameters may have limited impact compared to representation choices

Rather than interpreting small metric differences as meaningful, this project emphasizes the importance of **controlled experiments and cautious conclusions**.

The behavior of each model was similar in terms of metrics, with minor differences in amplitude, which can be explained by normal fluctuations.

![plot](/plots/val_loss_comparison.png)

Accuracy and F1, as well as Validation Loss, show a stoppage of growth at approximately 9 epochs with the default config.

![plot](/plots/default/Acc+F1_b512_d128_f100.png)

All other graphs for each model can be viewed in the `plots` directory of the repository.

---

## Comparison to Previous Work

In contrast to transformer-based experiments — where architectural and optimization changes often lead to visible effects — this project highlights that:

* increasing the depth (parameters) of the model does not always lead to improved results
* depth of the model significantly depends on:
  * assigned task
  * complexity of the dataset
  * data processing and presentation method

---

## Limitations

* single random seed per experiment
* limited set of hyperparameters explored
* a relatively simple and small dataset

---

## Key Takeaways

* Hyperparameter changes do not guarantee performance gains
* Small metric differences should be treated cautiously
* Controlled negative results are still valid experimental outcomes
* Simpler models may be more robust — and less sensitive — than expected

---

## Baseline performance

The following metrics were obtained on a separate test split of the dataset. The default model configuration was used for the sake of clarity

| Metric       | Score   | 
| ------------ | ------- |
| Average Loss | 0.298   |
| Accuracy     | 0.877   |
| Precision    | 0.889   |
| Recall       | 0.862   |
| F1           | 0.875   |

---
