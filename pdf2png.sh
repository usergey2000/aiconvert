#!/bin/bash
if [ "0$1" == "0" ]; then
    echo "Usage: ./$0 inputpdf"
    echo "Exmpl: ./$0 myreport.pdf"
    exit 0;
fi

inputpdf=$1
basefname="$(basename $inputpdf | sed 's/\./\ /g' | awk '{print $1}')"
echo $basefname
pdftoppm -png -r 150 $inputpdf ${basefname}-page
#
