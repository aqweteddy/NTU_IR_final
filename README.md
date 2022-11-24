# NTU IR Final Project 2022

## Competition

* [自然語言理解的解釋性資訊標記競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/26)

## data

* random split train / test: 0.85 / 0.15

## Method

* T5 conditional generation
* define 2 new tokens: `[r]`, `[q]`, represent start of `r` and `q`
* model input template: 
    * `[q] is <s> with [r]. [q] <q> [r] <r>`
    * `<q> <r> <s>` will be replaced with context.
    * `<s>` is the relation between `<q> and <r>`.
* model output template:
    * generate `r'`: `[r] ....`
    * generate `q'`: `[q] ....`
```py
python train.py
# see args in opt.py 
```

## Ckpt

### t5_v1.1-small
* template: `[q] is <s> with [r]. [q] <q> [r] <r>`
* step-5505
    * LCS score: 0.7982234334844397
