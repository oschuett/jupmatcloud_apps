#!/bin/bash

NAME=$1
NAME2="${NAME}-n10-l8-c3.5-g0.5_fastavg_nnrm"

set -x

#run glosim with landmark selection.
if [ ! -e "${NAME2}-nohead.oos.sim" ]; then
   #The output is a square matrix with the 300 selected landmarks and a rectangular matrix 2080x300.
   glosim.py ${NAME}.xyz -n 10 -l 8 -c 3.5 --kernel fastavg --distance  --periodic --nonorm --np 2 --nlandmarks 300
   #remove header to feed sketchmap and dimproj
   grep -v "#" "${NAME2}.landmarks.sim" > "${NAME2}.landmarks-nohead.sim"
   grep -v "#" "${NAME2}.oos.sim" > "${NAME2}-nohead.oos.sim"
fi

#run sketchmap on the landmarks
if [ ! -e "smap.landmark" ]; then
   ./sketch-map-auto.sh 300 "${NAME2}.landmarks-nohead.sim" "${NAME2}.landmarks" 0.039 3 3 0.039 3 3
   #remove header and 3rd column
   grep -v "#" "${NAME2}.landmarks.gmds" | awk '{print $1, $2}' > smap.landmark
fi

#project the dataset on the landmarks
if [ ! -e "${NAME2}.OOS.colvar" ]; then
    ./dimproj.sh dimproj 300 "${NAME2}.landmarks-nohead.sim" smap.landmark 0.039 3 3 0.039 3 3 "${NAME2}-nohead.oos.sim" "${NAME2}.OOS.colvar" 0.1 21 201 2> dimproj.log
fi

#EOF