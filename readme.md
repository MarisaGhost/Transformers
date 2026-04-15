# Assignment 3 Report: Multi-Class Text Classification Using Transformers

## Project Overview

This project fine-tunes transformer-based models for multi-class text classification on the 20 Newsgroups dataset. The main goal is not only to get a working classifier, but also to understand how preprocessing, augmentation, model architecture, regularization, interpretability, and deployment choices affect the final system.

The final selected 20 Newsgroups model is a BERT-based classifier with a fully connected classification head and classifier dropout of 0.3. The model was selected using validation macro F1, then evaluated once on the held-out test set. This keeps the model selection process cleaner because the test set is used for final reporting rather than repeated tuning.

## Dataset

The main dataset is 20 Newsgroups, which contains documents from 20 discussion categories such as computing, sports, politics, religion, science, vehicles, and marketplace posts. The project uses the official scikit-learn train/test split:

- Training source split: 11,314 documents
- Test split: 7,532 documents
- Number of classes: 20

The training portion was further split into train and validation sets with stratification:

- Train: 9,616 documents
- Validation: 1,698 documents
- Test: 7,532 documents

The notebook also includes AG News as a cross-domain transfer learning experiment. AG News has four labels: World, Sports, Business, and Sci/Tech. The AG News experiment is useful because Sports and Sci/Tech overlap somewhat with 20 Newsgroups, while World and Business are more different from the original domain.

## Preprocessing

Two preprocessing views were created.

The first view is transformer-friendly cleaning. It keeps the text fairly close to the original while removing noisy artifacts such as URLs, email patterns, punctuation, and repeated whitespace. This is the main input used for transformer fine-tuning.

The second view is a more classical NLP pipeline with stopword removal, lemmatization, and stemming. This was not assumed to be better. Instead, it was tested directly as an ablation to see whether aggressive traditional preprocessing helps or hurts a transformer model.

The OOV analysis showed a test OOV rate of 3.63% using a word-level vocabulary threshold. This matters for a traditional model, but it is less of a problem for BERT-style models because subword tokenization can split unknown words into known pieces. The result supports the decision to keep the main transformer input lightly cleaned instead of heavily normalized.

The preprocessing ablation confirmed this choice:

- Clean transformer text: test macro F1 = 0.6935
- Stopword/lemma/stem text: test macro F1 = 0.6263

The stronger result from the clean-text version suggests that the transformer benefited from keeping more of the original wording and context.

## Data Augmentation

The training set was expanded with three augmentation strategies:

- Synonym replacement
- Random word swap
- Back-translation

The final augmented training set increased from 9,616 to 10,034 examples, adding 418 samples. The augmentation mix was:

- Back-translation: 40 samples
- Random swap: 198 samples
- Synonym replacement: 180 samples

The purpose of augmentation was to improve robustness and reduce the effect of class imbalance. It was also used as part of the bias mitigation story, since minority classes benefit more from added examples.

## Model Architecture

The project compares three transformer backbones:

- DistilBERT
- BERT
- RoBERTa

It also compares three classification-head designs:

- Fully connected head
- Self-attention pooling head
- Multi-dropout head

The fully connected head uses the `[CLS]` representation and passes it through dropout, a hidden layer, ReLU, another dropout layer, and a final classification layer. The self-attention pooling head learns token-level pooling weights over the transformer output. The multi-dropout head applies several dropout rates and averages the classifier outputs during training.

This setup makes the comparison meaningful because the project does not only compare pretrained backbones. It also tests whether changing the classification head improves the final result.

## Training Setup

The training pipeline uses:

- AdamW optimizer
- Cosine learning-rate schedule with warmup
- Mixed precision on GPU
- Gradient accumulation
- Gradient clipping
- Early stopping
- Best-checkpoint saving based on validation macro F1

Validation macro F1 was used for model selection because the dataset has 20 classes and some classes are harder than others. Macro F1 gives each class equal weight, so it is more informative than accuracy alone.

## Experiment Results

### Backbone Comparison

BERT was selected as the best backbone by validation macro F1:

| Backbone | Test Accuracy | Test Macro F1 | Validation Macro F1 |
|---|---:|---:|---:|
| DistilBERT | 0.7037 | 0.6919 | 0.7382 |
| BERT | 0.7066 | 0.6935 | 0.7502 |
| RoBERTa | 0.6934 | 0.6783 | 0.7200 |

BERT had the best validation macro F1, so it was selected for later experiments.

### Classification-Head Comparison

The head comparison was run on the selected BERT backbone:

| Head | Test Accuracy | Test Macro F1 | Validation Macro F1 |
|---|---:|---:|---:|
| Fully connected | 0.7066 | 0.6935 | 0.7502 |
| Self-attention pooling | 0.7055 | 0.6933 | 0.7411 |
| Multi-dropout | 0.7116 | 0.6988 | 0.7349 |

The multi-dropout head had a slightly higher test macro F1, but the fully connected head had the best validation macro F1. Since the selection rule was validation-based, the fully connected head remained the selected model.

### Dropout Sweep

The dropout sweep tested classifier dropout values of 0.1, 0.3, and 0.5:

| Dropout | Test Accuracy | Test Macro F1 | Validation Macro F1 |
|---|---:|---:|---:|
| 0.1 | 0.7071 | 0.6935 | 0.7298 |
| 0.3 | 0.7066 | 0.6935 | 0.7502 |
| 0.5 | 0.6989 | 0.6829 | 0.7214 |

Dropout 0.3 was selected because it had the best validation macro F1. Dropout 0.5 appeared too strong for this setup and reduced performance.

### Final Selected Model

The final selected 20 Newsgroups model is:

- Backbone: BERT
- Head: Fully connected head
- Dropout: 0.3
- Test accuracy: 0.7066
- Test macro F1: 0.6935
- Weighted F1: 0.7064

## Per-Class Performance

The model performs best on clearer and more distinct categories. Some of the strongest classes include:

- `rec.sport.hockey`: F1 = 0.8807
- `rec.sport.baseball`: F1 = 0.8490
- `sci.med`: F1 = 0.8229
- `comp.windows.x`: F1 = 0.8213
- `talk.politics.mideast`: F1 = 0.8091

The weakest class is:

- `talk.religion.misc`: F1 = 0.2517

This class is difficult because it overlaps strongly with `soc.religion.christian`, `alt.atheism`, and political/religious discussion topics. The confusion matrix and misclassified examples make this clear.

## Confusion and Error Analysis

The largest error patterns were concentrated in semantically close groups:

- Religion categories were often confused with each other.
- Politics categories were often confused with religion or other politics categories.
- Computing hardware/software categories overlapped.
- `rec.autos` and `rec.motorcycles` were sometimes mixed.

The top confusion examples included:

- `talk.religion.misc` predicted as `talk.politics.guns`
- `talk.religion.misc` predicted as `soc.religion.christian`
- `talk.religion.misc` predicted as `alt.atheism`
- `rec.motorcycles` predicted as `rec.autos`
- `alt.atheism` predicted as `soc.religion.christian`

These mistakes are understandable because many posts contain shared vocabulary, quoted discussion, or mixed-topic arguments. For example, religion posts may discuss politics, abortion, or legal issues, while vehicle posts may mention general parts, engines, or sales.

## Bias Analysis

The bias analysis showed a wide spread in per-class F1:

- F1 range: 0.6290
- F1 standard deviation: 0.1463
- Worst class: `talk.religion.misc` with F1 = 0.2517
- Best class: `rec.sport.hockey` with F1 = 0.8807

There was also a strong class size to F1 correlation of 0.8779. This means that classes with more or clearer training examples tended to perform better. The notebook addresses this through minority-class augmentation and by reporting per-class metrics instead of hiding the issue behind overall accuracy.

The class balance ratio improved from 0.629 in the raw training split to 0.786 after augmentation. This does not completely remove class bias, but it is a concrete mitigation step.

## Interpretability

The project uses three interpretability methods:

- Attention visualization
- LIME
- SHAP

Attention plots show which tokens receive higher attention weights inside the transformer. These are helpful for qualitative inspection, but they should not be treated as a complete causal explanation.

LIME explains individual predictions by perturbing input text and estimating which words support or oppose the predicted label. For example, in an automobile-related document, features such as `bonnevilles`, `89`, `model`, and `se` supported the `rec.autos` prediction.

SHAP provides another explanation view by estimating feature contributions. Its results aligned with LIME in some cases, especially when class-specific terms were present.

Together, attention, LIME, and SHAP give a more complete explanation story than any one method alone.

## Cross-Domain Transfer Learning

The AG News transfer experiment compared two starting points:

- Generic pretrained BERT
- BERT encoder already fine-tuned on 20 Newsgroups

The 20 Newsgroups-tuned initialization performed better:

| AG News Starting Point | Validation Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| Generic pretrained BERT | 0.8718 | 0.8742 | 0.8678 |
| 20NG-tuned transfer | 0.8870 | 0.8925 | 0.8883 |

The transfer delta was +0.0205 test macro F1. This suggests that the 20 Newsgroups fine-tuning learned useful text classification features that transferred to AG News, especially because both datasets include technical and sports-related content.

## Deployment Plan

ONNX export was attempted, but both dynamic and static export failed with an `IndexError`. Instead of leaving deployment unfinished, the project now uses TorchServe as the real-time inference route.

The deployment architecture is:

```text
Dashboard -> FastAPI gateway -> TorchServe -> BERT classifier
```

The notebook includes a final deployment export section that saves:

- `model.pt`
- tokenizer files
- `config.json`
- `labels.json`
- `sample_requests.json`
- `deployment_readiness_report.md`
- `torchserve_commands.sh`

The TorchServe handler loads the saved model and tokenizer, accepts raw text input, applies the same cleaning logic, runs inference, and returns predicted classes with confidence scores.

The front-end dashboard provides a user-facing interface for real-time inference. It lets a user enter text, send it to the API, and view:

- Predicted class
- Confidence score
- Top class probabilities
- Model status
- Saved F1 and latency information
- Scalability notes

This makes the deployment more realistic than only saving a model file, because the model can be called from a real interface through an API.

## Scalability Considerations

The main scalability recommendation is batching. Transformer models are much more efficient when several inputs are processed together, so the API supports a `batch_size` field.

For low traffic, CPU serving may be acceptable. For higher traffic, GPU serving is recommended because BERT is relatively large.

The model uses a maximum sequence length of 256 tokens. This keeps memory and latency bounded, but it can truncate long documents. A production system should monitor how often inputs exceed the maximum length.

TorchServe can be scaled by increasing workers, running multiple replicas, and placing the FastAPI gateway behind a load balancer. Monitoring should include:

- Latency percentiles
- Throughput
- Error rate
- Input length distribution
- Predicted-class distribution
- Drift from the original training data

Artifacts should be versioned together:

- Model weights
- Tokenizer files
- Config file
- Label map
- Preprocessing rules
- TorchServe `.mar` archive

This makes rollback possible if a new model performs worse in production.

## Success Criteria Assessment

| Criterion | Status | Evidence |
|---|---|---|
| Multiple transformer models | Met | DistilBERT, BERT, and RoBERTa were fine-tuned and compared. |
| Classification head designs | Met | FC, self-attention pooling, and multi-dropout heads were implemented and compared. |
| Detailed evaluation | Met | Precision, recall, F1, per-class metrics, confusion matrix, and misclassified examples were reported. |
| Interpretability | Met | Attention visualization, LIME, and SHAP were included. |
| Bias analysis and mitigation | Met | Per-class F1 spread, class-size correlation, and minority-class augmentation were analyzed. |
| Cross-domain transfer | Met | AG News transfer learning improved macro F1 by 0.0205. |
| Deployment-ready model | Met after artifact export | TorchServe handler, FastAPI gateway, dashboard, artifact export, and scalability documentation were added. |
| Comprehensive reporting | Met | The notebook and this report document preprocessing, experimental design, findings, errors, and deployment. |

## Conclusion

The project successfully builds and evaluates a transformer-based multi-class classifier for 20 Newsgroups. BERT with a fully connected classification head and dropout 0.3 was selected using validation macro F1. The final test macro F1 was 0.6935, with strong performance on clearer classes such as hockey, baseball, medicine, and computing, and weaker performance on overlapping religion and politics categories.

The project also goes beyond basic model training by including preprocessing ablations, data augmentation, classification-head comparisons, dropout experiments, attention visualization, LIME, SHAP, bias analysis, AG News transfer learning, and a realistic deployment plan.

The main technical limitation is that ONNX export failed in the saved environment. The project addresses this by switching to TorchServe for real-time inference. With the final export cell, TorchServe handler, FastAPI gateway, and dashboard, the model has a clear production path once the trained artifact is exported.
