#!/bin/bash

cd cfexplainer

alphas=( "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" )

for alpha in "${alphas[@]}"
do
    for KM in {2..20..2}
    do
        echo "Running CFExplainer with alpha=$alpha, KM=$KM"

        python main.py \
            --do_test \
            --do_explain \
            --KM $KM \
            --cfexp_alpha $alpha \
            --cuda_id 0 \
            --hyper_para
    done
done


echo "==== ALL DONE ===="