#!/bin/bash
echo -n "enter prompt (e.g. a bento box): "
while read -r PROMPT && [ -z "$PROMPT" ]; do echo -n "enter prompt (e.g. a bento box): "; done
echo -n "enter negative prompt (e.g. low-res,blurry): "
while read -r N_PROMPT && [ -z "$N_PROMPT" ]; do echo -n "enter negative prompt (e.g. low-res,blurry): "; done

# read size
echo -n "enter image size (default: 512): "
while read -r SIZE
do
    # no input, default to 512
    if [[ -z $SIZE ]]
    then
	SIZE=512
	break
    fi

    # check if input is a positive integer
    if [[ ! $SIZE =~ ^[0-9]+$ ]]
    then
	echo "image size should be a positive integer"
    else
	break
    fi
    echo -n "enter image size (default: 512): "
done

if [[ -n "$PROMPT" ]];
then
    curl -X POST http://127.0.0.1:3000/txt2img -H 'Content-Type: application/json' -d "{\"prompt\":\"$PROMPT\",\"negative_prompt\":\"$N_PROMPT\",\"height\":$SIZE,\"width\":$SIZE}" --output output.jpg
else
    echo "No input"
fi
