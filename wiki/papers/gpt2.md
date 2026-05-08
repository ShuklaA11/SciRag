---
arxiv_id: gpt2
title: "Language Models are Unsupervised Multitask Learners"
generated_by: llama3.1:8b
generated_at: 2026-04-28T22:52:36+00:00
status: ok
---

# Language Models are Unsupervised Multitask Learners

## TL;DR
Language models can perform multiple tasks without explicit supervision when trained on a large dataset of webpages.

## Problem
Current machine learning systems are brittle and sensitive to changes in data distribution, requiring manual creation and labeling of datasets for each task.

## Method
The authors train language models on a large dataset of webpages called WebText, which enables the models to learn multiple tasks without explicit supervision. The largest model, GPT-2, is a 1.5B parameter Transformer that achieves state-of-the-art results on 7 out of 8 tested language modeling datasets in a zero-shot setting.

## Results
The authors achieve competitive and state-of-the-art results on various tasks, including question answering (55 F1 on the CoQA dataset) and machine translation, without using explicit supervision or task-specific architectures.

## Limitations
Unknown
