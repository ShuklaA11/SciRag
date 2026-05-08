---
arxiv_id: bert
title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
generated_by: llama3.1:8b
generated_at: 2026-04-28T22:51:50+00:00
status: ok
---

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## TL;DR
BERT is a pre-trained language model that achieves state-of-the-art results on eleven NLP tasks by using bidirectional representations.

## Problem
Current language representation models are unidirectional, limiting their power and restricting the choice of architectures for downstream tasks.

## Method
The authors propose BERT, which uses a 'masked language model' pre-training objective to enable deep bidirectional Transformer representations. This is achieved by randomly masking some tokens from the input and predicting the original vocabulary id based on context. Additionally, a 'next sentence prediction' task jointly pretrains text-pair representations.

## Results
BERT obtains new state-of-the-art results on eleven NLP tasks, including GLUE score (80.5%), MultiNLI accuracy (86.7%), SQuAD v1.1 question answering Test F1 (93.2), and SQuAD v2.0 Test F1 (83.1).

## Limitations
Unknown
