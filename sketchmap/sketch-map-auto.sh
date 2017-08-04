#!/bin/bash

# modify this variable so that it points to the path where the sketch-map
# executables are stored
SMAP="dimred"
#Dimentionality
HD=$1
LD=2
SIM=" -similarity "
DOW=""
DOT=""
PI=""
#File name of the similatity matrix
FILEHD=$2
#File name of the output projected 2D
FILELD=$3
# Sigma
SIGMAHD=$4
# a
AHD=$5
# b
BHD=$6

SIGMALD=$7
ALD=$8
BLD=$9


rm $FILELD.log
echo "Now running a preliminary iterative metric MDS and sketch-map."

if [ ! -e $FILELD.imds ]; then
   echo "$SMAP -vv -D $HD -d $LD $PI $DOW $DOT $SIM -center -preopt 100"
   grep -v \#  $FILEHD | $SMAP -vv -D $HD -d $LD $PI $DOW $DOT $SIM -center -preopt 100 > $FILELD.imds 2>>$FILELD.log
fi
grep -v "#" $FILELD.imds | awk '{print $1, $2}' > tmp
if [ ! -e $FILELD.ismap ]; then
   echo "$SMAP -vv -D $HD -d $LD $PI $DOW $DOT $SIM -center -preopt 100 -fun-hd $SIGMAHD,$AHD,$BHD -fun-ld $SIGMALD,$ALD,$BLD -init tmp"
   grep -v \#  $FILEHD | $SMAP -vv -D $HD -d $LD $PI $DOW $DOT $SIM -center -preopt 100 -fun-hd $SIGMAHD,$AHD,$BHD -fun-ld $SIGMALD,$ALD,$BLD -init tmp > $FILELD.ismap 2>> $FILELD.log
fi

GW=`awk 'BEGIN{maxr=0} !/#/{r=sqrt($1^2+$2^2); if (r>maxr) maxr=r} END{print maxr*1.2}' $FILELD.imds`;
NERR=`awk '/Error/{print $(NF)}'  $FILELD.imds | tail -n 1`
SMERR=`awk '/Error/{print $(NF)}'  $FILELD.ismap | tail -n 1`

IMIX=1.0
MAXITER=10
for ((ITER=1; ITER<=$MAXITER; ITER++)); do
   MDERR=$NERR
   echo "Mixing in $IMIX"
   if [ ! -e $FILELD.gmds_$ITER ]; then
     # echo "Now running $SMAP -vv -D $HD -d $LD $PI $DOW $DOT $SIM -center -preopt 50 -grid $GW,21,201 -fun-hd $SIGMAHD,$AHD,$BHD -fun-ld $SIGMALD,$ALD,$BLD -init tmp -gopt 3 -imix $IMIX < $FILEHD > $FILELD.gmds_$ITER 2>>log"
      grep -v \#  $FILEHD | $SMAP -vv -D $HD -d $LD $PI $DOW $DOT $SIM -center -preopt 50 -grid $GW,21,201 -fun-hd $SIGMAHD,$AHD,$BHD -fun-ld $SIGMALD,$ALD,$BLD -init tmp -gopt 3 -imix $IMIX > $FILELD.gmds_$ITER 2>>$FILELD.log
   fi
   grep -v "#" $FILELD.gmds_$ITER | awk '{print $1, $2}' > tmp
   GW=`awk 'BEGIN{maxr=0} !/#/{r=sqrt($1*$1+$2*$2); if (r>maxr) maxr=r} END{print maxr*1.2}' $FILELD.gmds_$ITER`;
   NERR=`awk '/Error/{print $(NF)}' $FILELD.gmds_$ITER  | tail -n 1 `
   echo "Residual error is $NERR"
   IMIX=`echo "$IMIX  $SMERR  $NERR" | awk '{new=$2/($2+$3); if (new<0.1) new=0.1; if (new>0.5) new=0.5; print new*$1 }'`
   echo "DEBUG  $MDERR $NERR"
   if [ ` echo $MDERR $NERR | awk -v i=$ITER '{ if (i>1 && (($1-$2)/$2)*(($1-$2)/$2)<1e-4) print "done"; else print "nope";}' ` = "done" ]; then ((ITER++)); break; fi;
done

echo "Doing final fit"
((ITER--))
grep -v "#" $FILELD.gmds_$ITER | awk '{print $1, $2}' > tmp
grep -v \#  $FILEHD | $SMAP -vv -D $HD -d $LD $PI $DOW $DOT $SIM -center -preopt 100 -grid $GW,21,201 -fun-hd $SIGMAHD,$AHD,$BHD -fun-ld $SIGMALD,$ALD,$BLD -init tmp -gopt 10 > $FILELD.gmds 2>>$FILELD.log


rm $FILELD.imds $FILELD.ismap $FILELD.gmds_*
rm global.*
rm tmp

#bash prep.sh $FILELD set
