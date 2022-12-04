#/bin/bash
for folder in $1/checkpoint*; 
do 
    echo $folder
     python eval.py --data ./data/val.csv --pretrained $folder --batch_size 32
done