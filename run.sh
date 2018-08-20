#!/bin/bash
if [ $# -ne 2 ] ; then
    echo "Illegal number of parameters"
    echo "Usage: run.sh <A file> <b file>"
    exit -1
fi
AFILE=$1
BFILE=$2
python3 lsqr.py "${AFILE}" "${BFILE}"
