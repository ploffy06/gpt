# gpt
This repo contains modified gpt models and the scripts to train them. These models have been trained on a small dataset of sonnets and so given a starting sequence,
the model will complete a sonnet.

# How to use
In your terminal, run

```
python3 rungpt.py {version} "{prompt}"
```
Where version is the gpt version you would like to use (e.g. v1, v2...) and
the prompt would be the starting sequence for your poem.

An example usage would be
```
python3 rungpt.py v1 "birds can't fly"
```

If you would like to not have a prompt (i.e. the model just generates it's own sonnet). Then,
```
python3 rungpt.py v1 ""
```

# About the Models
## GPTv1
This is a simple GPT model based on [Attention is All you Need](https://arxiv.org/abs/1706.03762)

## GPTv2
This is a modified version of the GPT in which the positional embedding has been added to the attention head instead.
This was based on [A Simple and Effective Positional Encoding for Transformers](https://arxiv.org/abs/2104.08698)
