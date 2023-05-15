#!/bin/bash
codedir="/cellar/users/jpark/code/dcell_mixed_precision/"
datadir="/cellar/users/jpark/code/dcell_mixed_precision/test/data"

inputfile="train_1.vector"
valfile="val_1.vector"
outputfile="train_1.log"

cudaid=$1

mkdir MODEL

python -u $codedir/train_dcell.py -onto $datadir/weissman_ontology_depth3.txt -gene2id $datadir/gene2id_mapping.txt -genotype_hiddens 5 -train $datadir/$inputfile -test $datadir/$valfile -cuda $cudaid > $outputfile
 
