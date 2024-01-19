#!/bin/bash

for file in pdf/*; do
  b=`basename $file .pdf`
  if [ -f png/$b.png ]; then
    if [ pdf/$b.pdf -nt png/$b.png ]; then
      NEEDS_CONVERSION=true
      echo \* $b.pdf changed, updating $b.png
    else
      NEEDS_CONVERSION=false
      echo $b.pdf unchanged
    fi
  else
    NEEDS_CONVERSION=true
    echo $b.png does not exist, creating
  fi
  if [[ $NEEDS_CONVERSION = true ]]; then
    convert -density 600 -background white -alpha remove -alpha off pdf/$b.pdf png/$b.png
  fi
done
