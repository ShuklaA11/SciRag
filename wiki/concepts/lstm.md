---
concept: "lstm"
generated_by: llama3.1:8b
generated_at: 2026-06-07T23:02:01+00:00
status: ok
depends_on: [1602.06291, 1603.07044, 1604.00117, 1604.00727, 1606.01404, 1606.03676, 1606.04631, elmo]
---

# lstm

## Definition
A type of Recurrent Neural Network (RNN) designed to handle long-term dependencies in sequential data.

## Origin
The concept of LSTM was first proposed in various papers, including one that introduced a Contextual LSTM model for improving performance on NLP tasks (1602.06291), and another that used an LSTM unit with attention mechanism to encode sentence pairs (1603.07044).

## Key Papers
- [[1602.06291]]
- [[1603.07044]]
- [[1604.00117]]
- [[1604.00727]]
- [[1606.01404]]
- [[1606.03676]]
- [[1606.04631]]
- [[elmo]]

## Variants
Notable variants include the bidirectional LSTM architecture (1604.00117, 1606.03676) and the character-level LSTM encoder (1604.00727). The BiLSTM framework was also proposed for video captioning tasks (1606.04631).

## Open Questions
The performance of LSTMs can vary across linguistic contexts, as seen in the ELMo model that uses bidirectional LSTM vectors trained on a large text corpus.
