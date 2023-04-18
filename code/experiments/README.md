# Dealing with the python message client

This script initiates the communication client and manages the corresponding AI reguests.

The tool was originally developed by Dr.-Ing. Marcus Grum.

## Getting-Started

### ...via message client remotely

1. Start the `message broker`. Further details can be found at the corresponding `Readme.md`.

1. Start the `messaging client` by

    ```
    python3 ../../messageClient/AI_simulation_basis_communication_client.py
    ```

1. Initiate `realize_annExperiment` request, which for instance can come from an Industry 4.0 production system, a modeling software or manually.

    ```
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=realize_annExperiment, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "localhost" -p 1883
    ```

### ...manually

1. Start the script by

    ```
    python3 experiment01.py
    ```

## Experiments

At each experiment, new knowledge bases are created.
Since knowledge bases shall be comparable over all experiment runs, subsequent experiments reuse knowledge bases that already have been set up in former experiments.
The following presents an overview of all knowledge bases, their initial creation, and their reuse.

<img src="../../documentation/KnowledgeBaseOverview.png" height="1440" />

### Experiment01

#### Proceeding for Experiment01

<img src="../../documentation/experiment01/Experiment01_Proceeding.png" height="600" />

#### KPIs in Experiment01

KPIs collected by the experiment can be found as follows:

<img src="../../documentation/experiment01/Experiment01_KpiCollection.png" height="600" />

#### Results of Experiment01

Results of this experiment can be found at the following path:

    ```
    repository/documentation/experiment01/
    ```

Here, one can find KPIs collected as well as plots generated.
For instance, the overview plot shows accuracies and losses of training and testing courses.

<img src="../../documentation/experiment01/plots/Plot_Average_Over_All_Experiments.png" height="1000" />

Here, one can see that bias is learnt successfully and forgotten successfully after learning base has changed.
After the change of the learning base, focused product type is learnt successfully.
Since learnt and unlearnt / forgotten knowledge base can be distinguished clearly separable (faced with non-overlapping standard deviations),
one can recognize this mechanism as effective approach to unlearn or intentionally forget in ANNs.
This can be seen at all, training and testing runs as well as accuracy and loss metrics.

A publication about this is in progress.

### Experiment02

#### Proceeding for Experiment02

<img src="../../documentation/experiment02/Experiment02_Proceeding.png" height="600" />

### Experiment03

#### Proceeding for Experiment03

<img src="../../documentation/experiment03/Experiment03_Proceeding.png" height="600" />

### Experiment01

#### Proceeding for Experiment04

<img src="../../documentation/experiment04/Experiment04_Proceeding.png" height="600" />