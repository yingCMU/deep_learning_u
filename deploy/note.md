## Machine Learning Workflow
1. Amazon Web Services (AWS) discusses their definition of the Machine Learning [Workflow](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-mlconcepts.html).

2. Google Cloud Platform (GCP) discusses their definition of the Machine Learning [Workflow](https://cloud.google.com/ai-platform/docs/ml-solutions-overview).

3. Microsoft Azure (Azure) discusses their definition of the Machine Learning [Workflow](https://docs.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-ml).


## Summary of Benefits of Risks Associated with Cloud Computing
The capacity utilization graph above was initially used by cloud providers like Amazon to illustrate the benefits of cloud computing. Summarized below are the benefits of cloud computing that are often what drives businesses to include cloud services in their IT infrastructure [1]. These same benefits are echoed in those provided by cloud providers Amazon (benefits), Google (benefits), and Microsoft (benefits).

Benefits
Reduced Investments and Proportional Costs (providing cost reduction)

Increased Scalability (providing simplified capacity planning)

Increased Availability and Reliability (providing organizational agility)

Below we have also summarized he risks associated with cloud computing [1]. Cloud providers don't typically highlight the risks assumed when using their cloud services like they do with the benefits, but cloud providers like: Amazon (security), Google (security), and Microsoft (security) often provide details on security of their cloud services. It's up to the cloud user to understand the compliance and legal issues associated with housing data within a cloud provider's data center instead of on-premise. The service level agreements (SLA) provided for a cloud service often highlight security responsibilities of the cloud provider and those assumed by the cloud user.

Risks
(Potential) Increase in Security Vulnerabilities

Reduced Operational Governance Control (over cloud resources)

Limited Portability Between Cloud Providers

Multi-regional Compliance and Legal Issues
References
1.   Erl, T., Mahmood, Z., & Puttini R,. (2013). Cloud Computing: Concepts, Technology, & Architecture. Upper Saddle River, NJ: Prentice Hall.

        Chapter 3: Understanding Cloud Computing provides an outline of the business drivers, benefits and risks of cloud computing.
Additional Resources
For the purpose of deploying machine learning models, it's important to understand the basics of cloud computing. The essentials are provided above, but if you want to learn more about the overall role of cloud computing in developing software, check out the [Optional] Cloud Computing Defined and [Optional] Cloud Computing Explained sections at the end of this lesson along with the additional resources provided below.

National Institute of Standards and Technology formal definition of Cloud Computing.
Kavis, M. (2014). Architecting the Cloud: Design Decisions for Cloud Computing Service Models. Hoboken, NJ: Wiley. Chapter 3 provides the worst practices of cloud computing which highlights both risks and benefits of cloud computing. Chapter 9 provides the security responsibilities by service model.
Amazon Web Services (AWS) discusses their definition of Cloud Computing.
Google Cloud Platform (GCP) discusses their definition of Cloud Computing.
Microsoft Azure (Azure) discusses their definition of Cloud Computing.

## Container & Docker
This container script is simply the instructions (algorithm) that is used to create a container; for Docker these container scripts are referred to as dockerfiles.
This is shown with the image below, where the container engine uses a container script to create a container for an application to run within. These container script files can be stored in repositories, which provide a simple means to share and replicate containers. For Docker, the [Docker Hub](https://hub.docker.com/search?q=&type=image) is the official repository for storing and sharing dockerfiles. Here's an [example](https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile) of a dockerfile that creates a docker container with Python 3.6 and PyTorch installed.

## Hyperparameters
If the machine learning platform fails to offer an automatic hyperparameter option, one option is to use methods from scikit-learn Python library for hyperparameter tuning. Scikit-learn is a free machine learning Python library that includes methods that help with [hyperparameter tuning](https://scikit-learn.org/stable/modules/grid_search.html#).

## Characteristics of Deployment
Model Versioning
One characteristic of deployment is the version of the model that is to be deployed.

Besides saving the model version as a part of a model’s metadata in a database, the deployment platform should allow one to indicate a deployed model’s version.
This will make it easier to maintain, monitor, and update the deployed model.

Model Monitoring
Another characteristic of deployment is the ability to easily monitor your deployed models.

Once a model is deployed you will want to make certain it continues to meet its performance metrics; otherwise, the application may need to be updated with a better performing model.
Model Updating and Routing
The ability to easily update your deployed model is another characteristic of deployment.

If a deployed model is failing to meet its performance metrics, it's likely you will need to update this model.
If there's been a fundamental change in the data that’s being input into the model for predictions; you'll want to collect this input data to be used to update the model.

The deployment platform should support routing differing proportions of user requests to the deployed models; to allow comparison of performance between the deployed model variants.
Routing in this way allows for a test of a model performance as compared to other model variants.

Model Predictions
Another characteristic of deployment is the type of predictions provided by your deployed model. There are two common types of predictions:

On-demand predictions
Batch predictions
On-Demand Predictions
On-demand predictions might also be called:
online,
real-time, or
synchronous predictions
With these type of predictions, one expects:
a low latency of response to each prediction request,
but allows for possibility high variability in request volume.
Predictions are returned in the response from the request. Often these requests and responses are done through an API using JSON or XML formatted strings.
Each prediction request from the user can contain one or many requests for predictions. Noting that many is limited based upon the size of the data sent as the request. Common cloud platforms on-demand prediction request size limits can range from 1.5(ML Engine) to 5 Megabytes (SageMaker).
On-demand predictions are commonly used to provide customers, users, or employees with real-time, online responses based upon a deployed model. Thinking back on our magic eight ball web application example, users of our web application would be making on-demand prediction requests.

Batch Predictions
Batch predictions might also be called:
asynchronous, or
batch-based predictions.
With these type of predictions, one expects:
high volume of requests with more periodic submissions
so latency won’t be an issue.
Each batch request will point to specifically formatted data file of requests and will return the predictions to a file. Cloud services require these files will be stored in the cloud provider’s cloud.
Cloud services typically have limits to how much data they can process with each batch request based upon limits they impose on the size of file you can store in their cloud storage service. For example, Amazon’s SageMaker limits batch predictions requests to the size limit they enforce on an object in their S3 storage service.
Batch predictions are commonly used to help make business decisions. For example, imagine a business uses a complex model to predict customer satisfaction across a number of their products and they need these estimates for a weekly report. This would require processing customer data through a batch prediction request on a weekly basis.


### Amazon Web Services (AWS) SageMaker
https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

- [Built-in Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) - There are at least fifteen built-in algorithms that are easily used within SageMaker. Specifically, built-in algorithms for discrete classification or quantitative analysis using linear learner or XGBoost, item recommendations using factorization machine, grouping based upon attributes using K-Means, an algorithm for image classification, and many other algorithms.
- Custom Algorithms - There are different programming languages and software frameworks that can be used to develop custom algorithms which include: [PyTorch](https://docs.aws.amazon.com/sagemaker/latest/dg/pytorch.html), TensorFlow, Apache MXNet, Apache Spark, and Chainer.
- [Your Own Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms.html) - Regardless of the programming language or software framework, you can use your own algorithm when it isn't included within the built-in or custom algorithms above.

-  SageMaker enables the use of [Jupyter Notebooks](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html) to explore and process data, along with creation, training, validation, testing, and deployment of machine learning models. This notebook interface makes data exploration and documentation easier.

- SageMaker provides a number of features and automated tools that make modeling and deployment easier:
  - [Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html): SageMaker provides a feature that allows hyperparameter tuning to find the best version of the model for built-in and custom algorithms. For built-in algorithms  SageMaker also provides evaluation metrics to evaluate the performance of your models.
  - [Monitoring Models](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-overview.html): SageMaker provides features that allow you to monitor your deployed models. Additionally with model deployment, one can choose how much traffic to route to each deployed model (model variant). More information on routing traffic to model variants can be found [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html) and [here](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html) .
  - Type of Predictions: SageMaker by default allows for [On-demand](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-test-model.html) type of predictions where each prediction request can contain one to many requests. SageMaker also allows for [Batch predictions](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-batch.html), and request data size limits are based upon S3 object size limits.

  ### Google Cloud Platform (GCP)
  Below we have highlighted some of the similarities and differences between these two cloud service platforms.

  - Prediction Costs: The primary difference between the two is how they handle predictions. With SageMaker predictions, you must leave resources running to provide predictions. This enables less latency in providing predictions at the cost of paying for running idle services, if there are no (or few) prediction requests made while services are running. With ML Engine predictions, one has the option to not leave resources running which reduces cost associated with infrequent or periodic requests. Using this has more latency associated with predictions because the resources are in a offline state until they receive a prediction request. The increased latency is associated to bringing resources back online, but one only pays for the time the resources are in use. To see more about ML Engine [pricing](https://cloud.google.com/ai-platform/training/pricing#node-hour) and SageMaker pricing.
  - Ability to Explore and Process Data: Another difference between ML Engine and SageMaker is the fact that Jupyter Notebooks are not available within ML Engine. To use Jupyter Notebooks within Google's Cloud Platform (GCP), one would use [Datalab](https://cloud.google.com/datalab/docs/). GCP separates data exploration, processing, and transformation into other services. Specifically, Google's Datalab can be used for data exploration and data processing, Dataprep can be used to explore and transform raw data into clean data for analysis and processing, and DataFlow can be used to deploy batch and streaming data processing pipelines. Noting that Amazon Web Services (AWS), also have data processing and transformation pipeline services like AWS Glue and AWS Data Pipeline.
  - Machine Learning Software: The final difference is that Google's ML Engine has less flexibility in available software frameworks for building, training, and deploying machine learning models in GCP as compared to Amazon's SageMaker. For the details regarding the two available software frameworks for modeling within ML Engine see below.

    - Google's TensorFlow is an open source machine learning framework that was originally developed by the Google Brain team. TensorFlow can be used for creating, training, and deploying machine learning and deep learning models. Keras is a higher level API written in Python that runs on top of TensorFlow, that's easier to use and allows for faster development. GCP provides both TensorFlow [examples](https://cloud.google.com/ai-platform/docs) and a [Keras](https://cloud.google.com/ai-platform/docs#census-keras) example.

    - Google's Scikit-learn is an open source machine learning framework in Python that was originally developed as a Google Summer of Code project. Scikit-learn and an [XGBoost Python package](https://xgboost.readthedocs.io/en/latest/python/index.html) can be used together for creating, training, and deploying machine learning models. In the in Google's [example](https://cloud.google.com/ai-platform/training/docs/training-xgboost), XGBoost is used for modeling and Scikit-learn is used for processing the data.
  - Flexibility in Modeling and Deployment: Google's ML Engine provides a number of features and automated tools that make modeling and deployment easier, similar to the those provided by Amazon's SageMaker. For the details on these features within ML Engine see below.
    - [Automatic Model Tuning](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview): Google's ML Engine provides a feature that enables hyperparameter tuning to find the best version of the model.
    - [Monitoring Models](https://cloud.google.com/ml-engine/docs/tensorflow/monitor-training): Google's ML Engine provides features that allow you to monitor your models. Additionally ML Engine provides methods that enable managing runtime [versions](https://cloud.google.com/ai-platform/training/docs/versioning) and managing [models and jobs](https://cloud.google.com/ai-platform/training/docs/managing-models-jobs).
    - Type of Predictions: ML Engine allows for [Online](https://cloud.google.com/ai-platform/prediction/docs/online-predict)(or On-demand) type of predictions where each prediction request can contain one to many requests. ML Engine also allows for [Batch predictions](https://cloud.google.com/ai-platform/prediction/docs/batch-predict). More information about [ML Engine's Online and Batch predictions](https://cloud.google.com/ai-platform/prediction/docs/online-vs-batch-prediction).

### Microsoft Azure
Similar to Amazon's SageMaker and Google's ML Engine, Microsoft offers Azure AI. Azure AI offers an open and comprehensive platform that includes AI software frameworks like: TensorFlow, PyTorch, scikit-learn, MxNet, Chainer, Caffe2, and other software like their Azure Machine Learning Studio.


# Building a mddel using sagemaker

## quota
There are three ways to view your quotas, as mentioned [here](https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html):
- Service Endpoints and Quotas,
- Service Quotas console, and
- AWS CLI commands - list-service-quotas and list-aws-default-service-quotas

In general, there are three ways to increase the quotas:
- Using Amazon Service Quotas [service](https://aws.amazon.com/about-aws/whats-new/2019/06/introducing-service-quotas-view-and-manage-quotas-for-aws-services-from-one-location/) - This service consolidates your account-specific values for quotas across all AWS services for improved manageability. Service Quotas is available at no additional charge. You can directly try logging into Service Quotas console [here](https://console.aws.amazon.com/servicequotas/home).
- Using AWS [Support Center](https://console.aws.amazon.com/support/home) - You can create a case for support from AWS.
- AWS CLI commands - request-service-quota-increase

You can view the Amazon SageMaker Service Limits at ["Amazon SageMaker Endpoints and Quotas"](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html) page. You can request to increase the AWS Sagemaker quota using the AWS Support Center
## creat notebook instance
https://www.youtube.com/watch?v=TRUCNy5Eqjc

# Bag of words
https://www.youtube.com/watch?v=A7M1z8yLl0w

https://en.wikipedia.org/wiki/Bag-of-words_model

## updating a deployed model.
Sometimes a model may not work as well as it once did due to changes in the underlying data. In this resource, you can read more about how a model's predictions and accuracy may degrade as a result of something called [concept drift](https://edouardfouche.com/Data-Stream-Generation-with-Concept-Drift/), which is a change in the underlying data distribution over time. When this happens we might want to update a deployed model, however, our model may be in use so we don't want to shut it down. SageMaker allows us to solve this problem without there being any loss of service.
## Creating and Using Endpoints
An endpoint, in this case, is a URL that allows an application and a model to speak to one another.

endpoint communicating between app and model.
### Endpoint steps
You can start an endpoint by calling .deploy() on an estimator and passing in some information about the instance.
`xgb_predictor = xgb.deploy(initial_instance_count = 1, instance_type = 'ml.m4.xlarge')`
Then, you need to tell your endpoint, what type of data it expects to see as input (like .csv).
```
from sagemaker.predictor import csv_serializer

xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
```
Then, perform inference; you can pass some data as the "Body" of a message, to an endpoint and get a response back!
```
response = runtime.invoke_endpoint(EndpointName = xgb_predictor.endpoint,   # The name of the endpoint we created
                                       ContentType = 'text/csv',                     # The data format that is expected
                                       Body = ','.join([str(val) for val in test_bow]).encode('utf-8'))

 ```
The inference data is stored in the "Body" of the response, and can be retrieved:
```
response = response['Body'].read().decode('utf-8')
print(response)
```
Finally, do not forget to shut down your endpoint when you are done using it.

`xgb_predictor.delete_endpoint()`

### Lambda function service
 It lets you perform actions in response to certain events, called triggers. Essentially, you get to describe some events that you care about, and when those events occur, your code is executed.

For example, you could set up a trigger so that whenever data is uploaded to a particular S3 bucket, a Lambda function is executed to process that data and insert it into a database somewhere.

One of the big advantages to Lambda functions is that since the amount of code that can be contained in a Lambda function is relatively small, you are only charged for the number of executions.

In our case, the Lambda function we are creating is meant to process user input and interact with our deployed model. Also, the trigger that we will be using is the endpoint that we will create using API Gateway.

To Create a Lambda Function:
1. Create an IAM Role for the Lambda function: Since we want the Lambda function to call a SageMaker endpoint, we need to make sure that it has permission to do so. To do this, we will construct a role that we can later give the Lambda function.
2. Create a Lambda function
```
# We need to use the low-level library to interact with SageMaker since the SageMaker API
# is not available natively through Lambda.
import boto3

# And we need the regular expression library to do some of the data processing
import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words

def bow_encoding(words, vocabulary):
    bow = [0] * len(vocabulary) # Start by setting the count for each word in the vocabulary to zero.
    for word in words.split():  # For each word in the string
        if word in vocabulary:  # If the word is one that occurs in the vocabulary, increase its count.
            bow[vocabulary[word]] += 1
    return bow


def lambda_handler(event, context):

    vocab = "*** ACTUAL VOCABULARY GOES HERE ***"

    words = review_to_words(event['body'])
    bow = bow_encoding(words, vocab)

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = '***ENDPOINT NAME HERE***',# The name of the endpoint we created
                                       ContentType = 'text/csv',                 # The data format that is expected
                                       Body = ','.join([str(val) for val in bow]).encode('utf-8')) # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    # Round the result so that our web app only gets '1' or '0' as a response.
    result = round(float(result))

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }
```
### API Gateway
At this point we've created and deployed a model, and we've constructed a Lambda function that can take care of processing user data, sending it off to our deployed model and returning the result. What we need to do now is set up some way to send our user data to the Lambda function.

The way that we will do this is using a service called API Gateway. Essentially, API Gateway allows us to create an HTTP endpoint (a web address). In addition, we can set up what we want to happen when someone tries to send data to our constructed endpoint.

In our application, we want to set it up so that when data is sent to our endpoint, we trigger the Lambda function that we created earlier, making sure to send the data to our Lambda function for processing. Then, once the Lambda function has retrieved the inference results from our model, we return the results back to the original caller.


- Some notes on Lambda and Gateway usage:
For Lambda functions you are only charged per execution, which for this class will be very few and still within the free tier. Deleting a lambda function is just a good cleanup step; you won't be charged if you just leave it there (without executing it). Similarly, for APIs created using API Gateway you are only charged per request, and the number of requests we require in this course should still fall under the free tier.

## Hyperparameter Tuning
SageMaker provides an automated way of doing this. In fact, SageMaker also does this in an intelligent way using Bayesian optimization. What we will do is specify ranges for our hyperparameters. Then, SageMaker will explore different choices within those ranges, increasing the performance of our model over time.

In addition to learning how to use hyperparameter tuning, we will look at Amazon's CloudWatch service. For our purposes, CloudWatch provides a user interface through which we can examine various logs generated during training.

# SageMaker Documentation
Developer Documentation can be found here: https://docs.aws.amazon.com/sagemaker/latest/dg/

Python SDK Documentation (also known as the high level approach) can be found here: https://sagemaker.readthedocs.io/en/latest/

Python SDK Code can be found on github here: https://github.com/aws/sagemaker-python-sdk
