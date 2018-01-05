#!/bin/bash
COMMAND= python webapp.py --port 5000 --model porn_model --type RCNN --pullrequest &
#COMMAND=ls
#PORT NUMBER
PORT=5000
# path to the log file
GPU_LOGFILE=/home/ubuntu/gpu_log.txt
TOP_LOGFILE=/home/ubuntu/top_log.txt
CURL_FILE=/home/ubuntu/curl_log.txt
#interval
INTERVAL=0.1


nohup $COMMAND &
#STORING PID
PID=$!


echo Script started with process id:$PID

while true;
do
        nvidia-smi >> $GPU_LOGFILE;
        echo '--------------------------------------------------------------------------------------------' >> $GPU_LOGFILE;
        echo  >> $GPU_LOGFILE;
        top -n 1 -p $PID -b >>  $TOP_LOGFILE;
        echo '--------------------------------------------------------------------------------------------' >> $TOP_LOGFILE;
        echo  >> $TOP_LOGFILE;
        netstat -a | grep ESTABLISHED | grep -c :$PORT >> $CURL_FILE;
        echo '--------------------------------------------------------------------------------------------' >> $CURL_FILE;
        echo  >> $CURL_FILE;

        sleep $INTERVAL;
done

