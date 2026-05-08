---
arxiv_id: elmo
title: "Deep contextualized word representations"
generated_by: llama3.1:8b
generated_at: 2026-04-28T22:52:11+00:00
status: ok
---

# Deep contextualized word representations

## TL;DR
A new type of word representation called ELMo is introduced, which models complex characteristics of word use and how they vary across linguistic contexts.

## Problem
Traditional pre-trained word representations only allow a single context-independent representation for each word, failing to model polysemy and complex characteristics of word use.

## Method
ELMo uses vectors derived from a bidirectional LSTM that is trained with a coupled language model objective on a large text corpus. The internal states of the biLM are combined in a linear manner to create rich word representations.

## Results
ELMo significantly improves the state of the art across six challenging NLP problems, including question answering and sentiment analysis, with up to 20% relative error reductions.

## Limitations
Unknown
