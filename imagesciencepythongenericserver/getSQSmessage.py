import _init_paths
#import boto.sqs
import boto3
from boto3.session import Session
import botocore
import argparse

#import webapp ###bad

''' This establishes the SQS connection (where all the incomming URls are stored)
    and returns the JSON.
'''

if _init_paths.args['pullrequest'] == True:
    # Create the Boto3 Session
    #3pKJeujER8OMGeHKnW5AkgrOj+ASFS+jtSSbH1wc
    #AKIAJRB7UETVGEPPOYVQ

    session = Session(aws_access_key_id='', aws_secret_access_key='', region_name='us-east-1')

    #Get service resource
    client = session.client('sqs')

    # Get the queue. This returns an SQS.Queue instance
    #response = client.get_queue_url(QueueName = 'vertex-main')
    response = client.get_queue_url(QueueName = '')

    url = response['QueueUrl']
else:
    print " not vertex"

def getMessage():
    messages = client.receive_message( QueueUrl=url, AttributeNames=['All'], MaxNumberOfMessages=1)
    if messages.get('Messages'):
        m = messages.get('Messages')[0]
        body = m['Body']
        receipt_handle = m['ReceiptHandle']
    else:
        body = ""
        receipt_handle = ""

    return (body, receipt_handle)

def deleteMessage(receipt_handle):
    #print type(receipt_handle)           #testing
    #print str(receipt_handle)            #testing
    resultMessage = ""
    try:
        resp = client.delete_message(QueueUrl = url, ReceiptHandle = receipt_handle)
        #print resp
        resultMessage = "Message successfully deleted"
    except botocore.exceptions.ClientError:
        resultMessage = "Error deleting message..Message is still in queue."

    return resultMessage
