#!/bin/bash
hmDir=$PWD

runno=1

for temp in "35.0" "40.0" "45.0" "50.0" "55.0" "60.0" "65.0" "70.0";do
#for temp in "47.0" "48.0" "49.0" "51.0" "52.0" "53.0" "54.0" "56.0" "57.0" "58.0" "59.0" "61.0" "62.0";do
#for temp in "36.0" "37.0" "38.0" "39.0" "42.0" "46.0";do
#for temp in "63.0" "64.0" "66.0" "67.0" "68.0" "69.0";do

$hmDir/WHAM_Grossfield/wham/wham/wham -0.5 60.5 61 1e-4 ${temp} 0 T_${temp}_run_${runno}_metadata.dat Free_Energy/free_energy_T_${temp}_run_${runno}.txt	

done
