#!/bin/bash

if [[ ! -d real_data ]]; then
    mkdir -p real_data
fi

wget -O real_data/IMPC_ProcessedData_Continuous.csv https://zenodo.org/record/2396572/files/IMPC_ProcessedData_Continuous.csv
