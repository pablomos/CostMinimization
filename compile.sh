#!/bin/bash

python3 -c "import numpy"
if [ $? -eq 0 ] ; then
    echo "NumPy is installed"
else
    python3 -c "import pip"
    if [ $? -eq 0 ] ; then
	echo "Pip is installed"
    else
	python3 get-pip.py --user
    fi
    python3 -m pip install --user numpy
fi

python3 -c "import scipy"
if [ $? -eq 0 ] ; then
    echo "SciPy is installed"
else
    python3 -c "import pip"
    if [ $? -eq 0 ] ; then
	echo "Pip is installed"
    else
	python3 get-pip.py --user
    fi
    python3 -m pip install --user scipy
fi

echo "INFO: completed the installation. Proceed to running run.sh"
