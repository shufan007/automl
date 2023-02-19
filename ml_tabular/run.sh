#!/usr/bin/env bash

source ~/.bashrc
export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`

export PATH="/home/luban/miniconda3/bin:$PATH"
conda activate base

export PYTHONPATH=./

if [[ $1 =~ ".py" ]]; then
  python $@
elif [[ $1 =~ "autotabular" ]]; then
  $@
elif [[ $3 =~ ".py" ]]; then
  export HADOOP_USER_NAME=$1
  export HADOOP_USER_PASSWORD=$2

  python ${@:3}
elif [[ $3 =~ "autotabular" ]]; then
  ${@:3}
fi
