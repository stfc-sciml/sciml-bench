#!/bin/bash
# Check if aws is installed
if ! command -v aws &> /dev/null
then
echo “AWS is not installed.”
echo “To install AWS run: pip install aws-shell”
exit
fi
if [ $# -ne 2 ]
then
echo “Please provide path to data server and to target directory.”
echo “For example: ./downloadData.sh s3://sciml-datasets/ms/em_graphene_sim ./graphene
exit
fi
ENDPOINT=https://s3.echo.stfc.ac.uk
aws s3 –no-sign-request –endpoint-url $ENDPOINT sync $1 $2
echo “Dataset download complete”
