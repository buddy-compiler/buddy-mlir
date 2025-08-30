#!/bin/bash
FILTER_min=$1
FILTER_max=$2
FILTER_step=$3
OUTPUT_min=$4
OUTPUT_max=$5
OUTPUT_step=$6
if [ ! -d tempFile  ];then
  mkdir tempFile
fi

if [ ${OUTPUT_step} -eq 0 ]; then
    for((FILTER=${FILTER_min};FILTER<=${FILTER_max};FILTER+=${FILTER_step}));do
        for((OUTPUT=${OUTPUT_min};OUTPUT<=${OUTPUT_max};OUTPUT*=2));do
            FILE=tempFile/conv-$FILTER-$OUTPUT.mlir
            cp conv2d-buddy-template $FILE
            if [ "$(uname)" == "Darwin" ]; then
                sed -i '' 's/TEMPLATE_FILTER/'$FILTER'/g' $FILE
                sed -i '' 's/TEMPLATE_OUTPUT/'$OUTPUT'/g' $FILE
            elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
                sed -i 's/TEMPLATE_FILTER/'$FILTER'/g' $FILE
                sed -i 's/TEMPLATE_OUTPUT/'$OUTPUT'/g' $FILE
            fi
        done;
    done;
else
    for((FILTER=${FILTER_min};FILTER<=${FILTER_max};FILTER+=${FILTER_step}));do
        for((OUTPUT=${OUTPUT_min};OUTPUT<=${OUTPUT_max};OUTPUT+=${OUTPUT_step}));do
            FILE=tempFile/conv-$FILTER-$OUTPUT.mlir
            cp conv2d-buddy-template $FILE
            if [ "$(uname)" == "Darwin" ]; then
                sed -i '' 's/TEMPLATE_FILTER/'$FILTER'/g' $FILE
                sed -i '' 's/TEMPLATE_OUTPUT/'$OUTPUT'/g' $FILE
            elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
                sed -i 's/TEMPLATE_FILTER/'$FILTER'/g' $FILE
                sed -i 's/TEMPLATE_OUTPUT/'$OUTPUT'/g' $FILE
            fi
        done;
    done;
fi
