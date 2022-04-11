#!/bin/bash
jobID=`qsub -j y -cwd -q gpu.q -l gpu=1 run.sh train_text2text.py --batch-size 16 --epochs 10 --checkpoint google/mt5-small`

#jobID=`echo $jobID | awk 'match($0,/[0-9]+/){print substr($0, RSTART, RLENGTH)}'`

#sleep 10s
#tail -f run.sh.o$jobID