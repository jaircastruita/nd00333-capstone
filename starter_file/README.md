# Employee attrition prediction project

This project consist in comparing and selecting 2 different approaches to solve a usual problem of best model selection for certain dataset. The dataset selected for this project will be the *employee attrition* dataset, a toy dataset made by the IBM data scientists with the purpose of predictor's explanation. For more info about this dataset you can follow this [link](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).

The workflow of this project will be the following: First upload the dataset of employee attrition into the azureml workspace. Use hyperdrive (fixed model/moving hyperparameters) and automl (moving model/moving hyperparameters) with the data attrition dataset and compare the best performing models resulting of both azure ml tools. Register, deployment and consumption of such model will be followed once the best performing model is identified.

![capstone image](images/capstone-diagram.png)

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview
The dataset used for this project is the employee attrition dataset, from kaggle; The data in this dataset is completely simulated, so there is no violation to individual's right to privacy.

### Task

This dataset is formed of 35 columns: 26 integer columns, 6 categorical columns and 3 binary columns which simulates traits for each employee. Each row represents a different employee and the main objective is to find relations between the dependent variable (**Attrition**) and the remaining columns. For this project the main purpose is to build a model that predicts Attrition given a predictor's vector and maybe identify which columns have more influence in the predicted result.

### Access

First, the dataset load is made by using the azureml TabularDataset method *from_delimited_files* and passing the CSV file path as an argument. This cell is present in both notebooks.

```python
data_path = "https://raw.githubusercontent.com/jaircastruita/nd00333-capstone/master/starter_file/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

ibm_ds = Dataset.Tabular.from_delimited_files(path=data_path)
```

After executing this cell a TabularDataset is returned. This dataset will be necessary when training and consuming our future model and model as a service.

## Automated ML

Given the tabular nature of the selected dataset with categorical and numeric columns present, relatively few observations with respect the number of columns, the following settings were selected in the automl configuration:

- *iteration_timeout_minutes* to 20, just to make sure the experiment will finish in a feasible time
- *enable_early_stopping* to True in order to not waste resources when the model performance is not improving anymore
- *primary_metric* "AUC_weighted" given we are facing an umbalanced class problem. click [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) for more info.
- *verbosity* logging.INFO to capture important events for the model experiments
- *max_concurrent_iterations* to 5 in order to parallelize experimentation
- *n_cross_validations* to 5 to give an estimated 80-20 proportion to the training-testing sets in each fold. Though, maybe more folds could be suggested due the lack of observations in the dataset
- *task* will be "classification" because the objective is to identify attrition as a "Yes", "No" labeling problem 

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

After automl is done with model/hyperparameter search, the model with the best performance metric is selected. The results obtained with automl outperforms the ones found using *hyperdrive* so, for the deployment section the automl model will be used.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Due the hyperparameter tuning section suposes a fixed model, a PyTorch DNN with 2 layers was selected. The selection of a PyTorch model was just for didactic purpose, using it as an exercise to work defining a proper Environment class along with a good sklearn preprocess estimator. This also is a wonderful moment to try a compute cluster that actually supports GPU computing. For hyperparameter optimization the following settings were selected:

- *num_epochs* to determine the training cycles for the DNN in order to achieve the best performance and to not overfit
- *learning_rate* is the initial learning hyperparameter in order to reduce the grandient values by a small leap factor (here, we are using Adam optimizer. This substracts importance to the search of this term but nonetheless is a useful point to start anyway)
- *num_layer1* is for determining the neuron number the first layer of the DNN
- *num_layer2* is to determine the second layer of the DNN neuron number

```python
run.log("num_epochs", np.int(num_epochs))
run.log("learning_rate", np.float(learning_rate))
run.log("num_layer1", np.int(num_layer1))
run.log("num_layer2", np.int(num_layer2))
```

The above cell is used to log the hyperparameters mentioned above.

In terms of performance metric we are using the same that was used for the automl experiment.

```python
run.log('best_test_acc', np.float(best_metric))
```

The above cell reports the best metric per epoch with the label *best_test_acc*.

For hyper-parameter search, it is defined in the following code snippet. We use *BayesianParameterSampling* for parameter sampling. For this specific sampling it's not needed to define a stop policy. The default arguments are applied. Due the nature of the selected hyper-parameters to search de following ranges were selected:

```python
param_sampling = BayesianParameterSampling({
    "learning_rate": uniform(0.001, 0.1),
    "num_epochs": choice(range(100, 5000)),
    "layer1": choice(range(10, 100)),
    "layer2": choice(range(10, 100)),
})
```

Note that *uniform* is for uniform float distribution. For integer distributions the *choice* funtion with the *min* to *max* range is passed.

### Results

It is very important to note that some aspects of the dataset were neglected during this run, such as class imbalance. There exist many different approaches to solve this kind of problems like undersampling or oversampling methods (maybe including a specialized library for this task such as [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)) or using a weighted loss function but for now it is not the scope of this project.

The results reported by the hyperdrive run experiment were the following:

- num_epochs: 
- learning_rate:
- num_layer1:
- num_layer2:
- best_test_acc:

Although the results were good the experiment was surpassed by the automl result with an *AUC_weighted* metric of **VALOR**. For that reason, the model selected for deployment will be the resulting from the automl selection process. Please inspect the *hyperparameter_tuning.ipynb* for more details.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Once the best performing model has been selected we can move on to deploy it. This part will be present in the *automl.ipynb* due its superior result.

### Registering the model

We first have to retrieve the best run/model from the AutoMLRun object. For this use use the *get_outputs()* method:

```python
best_automl_run, best_automl_model = remote_run.get_output()
```

Next download the inference script generated by the best model. The last line of the next cell denotes that we are downloading the best model script in the scoring path used at inference time.

```python
model_name = best_automl_run.properties["model_name"]
script_file_name = "inference/score.py"
best_automl_run.download_file('outputs/scoring_file_v_1_0_0.py', script_file_name)
```

```python
description = 'sample service for Automl Classification'
model = best_automl_run.register_model(description = description,
                                       model_name=model_name,
                                       tags={'area': "Attrition", 'type': "automl_classification"},
                                       model_path = "outputs/model.pkl")

# Combine scoring script & environment in Inference configuration
inference_config = InferenceConfig(entry_script=script_file_name)

# Set deployment configuration
deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, 
                                                       memory_gb=1,
                                                       tags={'area': "AttritionData", 'type': "automl_classification"},
                                                       description='sample service for Automl Classification')

# Define the model, inference, & deployment configuration and web service name and location to deploy
service = Model.deploy(
    workspace=ws,
    name="automl-web-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config)

service.wait_for_deployment(show_output=True)
```

In the second line we register the model giving as arguments some necesary data such as the model name, en optional data such as description. Next we instantiate a InferenceConfig class which combines the scoring script and the environment (if provided) with the inference configuration. For this deployment we use an Azure Container Instance passing the number of cores and memory that will be used by the inference compute instance as the model *endpoint*. We also pass some optional arguments such as *tags* and *description*.

Last, a *Model* class is instantiated and deployed passing in the model, the inference and deployment configuration along with the name of the service created. Once done we'll end up with the deployed endpoint to send some queries and try it. To see this section please refer to the last section of the *automl.ipynb* file.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
