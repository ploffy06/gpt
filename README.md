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

```
train loss: 1.7585, val loss: 2.0922
```

## GPTv2
This is a modified version of the GPT in which the positional embedding has been added to the attention head instead.
This was based on [A Simple and Effective Positional Encoding for Transformers](https://arxiv.org/abs/2104.08698)

```
step 2990: train loss: 0.8103, val loss: 2.3866
```

## GPTv3
Although `v2` achieved a much smaller train loss, it suffers greatly from the overfitting problem. One method to resolve this was to train it on
1500 epochs as opposed to 3000 (i.e. early stopping. The choice of 1500 epochs was done by inspection of `results/rest_v2.txt`).

```
train loss: 1.5006, val loss: 1.9767
```
## GPTv4
However, we want a more nuanced solution than changing the number of epochs.

```
train loss: 0.8129, val loss: 2.3534
```
