#!/bin/bash

PATH2dimproj=$1

# number of landmark 
dim=$2
# file name of landmark similarity matrix
P2landsim=$3
# file name of landmark CV1 CV2
P2land2d=$4
# sigmoid parameters
sigmahd=$5
A=$6
B=$7
sigmald=$8
a=$9
b=${10}
# file name of out of sample rectagular similarity matrix
P2OOSsim=${11}
# output file name
P2OOS2d=${12}
# extreme values of the landmark sketchmap
range=${13}
# nb of discretization grid points on 2 levels (21 and 201 typicaly)
coarseGrid=${14}
fineGrid=${15}


$PATH2dimproj -D $dim -d 2 -P $P2landsim -p $P2land2d -fun-hd $sigmahd,$A,$B -fun-ld $sigmald,$a,$b -grid $range,$coarseGrid,$fineGrid -cgmin 5 -similarity < $P2OOSsim > $P2OOS2d

