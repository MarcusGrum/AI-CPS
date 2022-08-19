# Dealing with the python message client

This script initiates the communication client and manages the corresponding AI reguests.

The tool was originally developed by Dr.-Ing. Marcus Grum.

## Getting-Started

### ...via message client remotely

1. Start the `message broker`. Further details can be found at the corresponding `Readme.md`.

1. Start the `messaging client` by

    ```
    python3 ../messageClient/AI_simulation_basis_communication_client.py
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

### Experiment01

#### Proceeding

<img src="../../documentation/ExperimentProceeding.png" height="600" />

#### KPIs

KPIs collected by the experiment can be found as follows:

<img src="../../documentation/ExperimentKpiCollection.png" height="600" />