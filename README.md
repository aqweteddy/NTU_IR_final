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

### t5_v1.1-base

* template: `[q] is <s> with [r]. [q] <q> [r] <r>`
* step: 22026
    * 0.80398513787775
* step: 36710
    * 0.8098746462247618
    * public test: 0.796762
    
### BART_base

* template: `[q] is <s> with [r]. [q] <q> [r] <r>`
* public test: 0.772862
* step 5505
    * 0.7837674902359171
* step 7340
    * 0.7880135306152085

### BART_base_CNN

* step 5505
    * 0.7933196511139557
* step 7340
    * 0.7935548323819159