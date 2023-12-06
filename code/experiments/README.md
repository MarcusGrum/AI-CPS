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

The `experiment01` simulates the manipulation of CPS knowledge base by process change - alternative product change (in context of continual learning and training data manipulation): 
Inhowfar do AI-based CPS forget if a process changes leads to a different product type distribution (in this case alternative products) that does not correspond to the CPS's current specialization?

#### Proceeding for Experiment01

<img src="../../documentation/experiment01/Experiment01_Proceeding.png" height="600" />

#### KPIs in Experiment01

KPIs collected by the experiment can be found as follows:

For training as well as testing, `accuracies`, `losses`, `number of data` (training or testing data) have been collected as individual kpi files.
These have been reorganized and summarized as `mu`, `sigma` and `n` as the following presents.

<img src="../../documentation/experiment01/Experiment01_KpiCollection.png" height="600" />

These files and KPIs are used for statistical analyses, whose code can be found at the following path:

    ```
    repository/documentation/experiment01/statistics
    ```

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

Statistics proofs this on a significant level.

#### Attempts for a research answer of Experiment01

The research question can be answered as follows:
If a process change leads to a different product type distribution (in this case alternative products) that does not correspond to the CPS's current specialization,
the CPS's current specialization adapts to the currently required set of skills
and fully forgets once learnt knowledge effectively while relevant knowledge (as required by the new process) is learnt.

On the other hand, the corresponding data set manipulation shows as adequate mechanism to fully forget knowledge:
The one kind of knowledge shall be forgotten, while the other kind of knowledge shall be learnt.

A publication about this is in progress.

### Experiment02

The `experiment02` simulates the manipulation of CPS knowledge base by smart sensors overtaking tasks partly - filtering sensory input (in context of continual learning and training data manipulation): 
Inhowfar do AI-based CPS forget if fruit evaluation is overtakten by preceeding smart sensors and the CPS's current specialization does not correspond to the required set of skills?

#### Proceeding for Experiment02

<img src="../../documentation/experiment02/Experiment02_Proceeding.png" height="600" />

#### KPIs in Experiment02

KPIs collected by the experiment can be found as follows:

For training as well as testing, `accuracies`, `losses`, `number of data` (training or testing data) have been collected as individual kpi files.
These have been reorganized and summarized as `mu`, `sigma` and `n` as the following presents.

<img src="../../documentation/experiment02/Experiment02_KpiCollection.png" height="600" />

These files and KPIs are used for statistical analyses, whose code can be found at the following path:

    ```
    repository/documentation/experiment02/statistics
    ```

#### Results of Experiment02

Results of this experiment can be found at the following path:

    ```
    repository/documentation/experiment02/
    ```

Here, one can find KPIs collected as well as plots generated.
For instance, the overview plot shows accuracies and losses of training and testing courses.

<img src="../../documentation/experiment02/plots/Plot_Average_Over_All_Experiments.png" height="1000" />

Here, one can see that bias and manipulation are learnt successfully and forgotten as well as remained successfully after learning base has changed.
After the change of the learning base, bias focused product type is remained successfully, while mainipulation product type is forgotten successfully.
Since learnt and unlearnt / forgotten knowledge base can be distinguished clearly separable (faced with non-overlapping standard deviations),
one can recognize this mechanism as effective approach to unlearn or intentionally forget in ANNs.
This can be seen at all performance levels, which means all training and testing runs as well as accuracy and loss metrics.

Statistics will proof this on a significant level.

#### Attempts for a research answer of Experiment02

The research question can be answered as follows:
If preceeding smart sensors overtake tasks partly, the CPS's current specialization adapts to the currently required set of skills
and preserves once learnt knowledge effectively while irelevant knowledge (as the smart sensor cares about this) is forgotten.

On the other hand, the corresponding data set manipulation shows as adequate mechanism to partly forget knowledge:
The one kind of knowledge shall be preserved, while the other kind of knowledge shall be forgotten.

A publication about this is in progress.

### Experiment03

The `experiment03` simulates the manipulation of CPS knowledge base by worsening sensors badly affecting tasks: filtering sensory input (in context of continual learning and training data manipulation): 
Inhowfar do AI-based CPS forget if current evaluation task is disturbed by defect sensors, such as providing blurred images that do not correspond to the CPS's current specialization?

#### Proceeding for Experiment03

<img src="../../documentation/experiment03/Experiment03_Proceeding.png" height="600" />

#### KPIs in Experiment03

KPIs collected by the experiment can be found as follows:

For training as well as testing, `accuracies`, `losses`, `number of data` (training or testing data) have been collected as individual kpi files.
These have been reorganized and summarized as `mu`, `sigma` and `n` as the following presents.

<img src="../../documentation/experiment03/Experiment03_KpiCollection.png" height="600" />

These files and KPIs are used for statistical analyses, whose code can be found at the following path:

    ```
    repository/documentation/experiment03/statistics
    ```

#### Results of Experiment03

Results of this experiment can be found at the following path:

    ```
    repository/documentation/experiment03/
    ```

Here, one can find KPIs collected as well as plots generated.
For instance, the overview plot shows accuracies and losses of training and testing courses.

<img src="../../documentation/experiment03/plots/Plot_Average_Over_All_Experiments.png" height="1000" />

Here, faced with the comparison of `clean refinement` and `disturbance`, one can see 
that the specialization shows a certain fault tolerance.
Although the disturbed input shows worse performance characteristics (only in terms of `accuracy`, but not in terms of `loss`),
even disturbed input shows acceptable performance.

Faced with the comparions of `bias` with `manipulation`, one can see
that a focus on learning disturbed material will lead to a slightly better performance in this disturbed setting.
But if tends to worsen performance on the biased, clean and original setting.

Statistics will proof this on a significant level.

#### Attempts for a research answer of Experiment03

The research question can be answered as follows:
If defect sensors worsen input, the CPS's current specialization will adapt to the worsened setting and the currently required set of skills.
This might be interpreted to forgetting for a certain percentage.
As the CPS as learned to deal with the defect input, performance will rise in defect environment setting, but lower performance in the original, non-defect environment setting.
So, before a forgetting is trained in an expensive way, it is recommended to repair the sensor at a reasonable price.

On the other hand, the corresponding data set manipulation shows as adequate mechanism to forget at a percentage knowledge level:
The one kind of knowledge will be preserved, while fine details of that kind of knowledge will be forgotten.

A publication about this is in progress.

### Experiment04

The `experiment04` simulates the manipulation of CPS knowledge base by process change - sub-set product change (in context of continual learning and training data manipulation): 
Inhowfar do AI-based CPS forget if a process changes leads to a different product type distribution (in this case a sub-set of trained products) that does not correspond to the CPS's current specialization?

#### Proceeding for Experiment04

<img src="../../documentation/experiment04/Experiment04_Proceeding.png" height="600" />

#### KPIs in Experiment04

KPIs collected by the experiment can be found as follows:

For training as well as testing, `accuracies`, `losses`, `number of data` (training or testing data) have been collected as individual kpi files.
These have been reorganized and summarized as `mu`, `sigma` and `n` as the following presents.

<img src="../../documentation/experiment04/Experiment04_KpiCollection.png" height="600" />

These files and KPIs are used for statistical analyses, whose code can be found at the following path:

    ```
    repository/documentation/experiment04/statistics
    ```

#### Results of Experiment04

Results of this experiment can be found at the following path:

    ```
    repository/documentation/experiment04/
    ```

Here, one can find KPIs collected as well as plots generated.
For instance, the overview plot shows accuracies and losses of training and testing courses.

<img src="../../documentation/experiment04/plots/Plot_Average_Over_All_Experiments.png" height="1000" />

Here, one can see 
that the `bias` worsens when training focuses on a sub-set of that training material.
Apparently, the complement has been unlearnt efficiently.

The `manipulation` improves slightly due to the refinement on exactly the manipulation material.
The `complement` worsens drastically due to its exclusion in learning procedures. 

Statistics will proof this on a significant level.

#### Attempts for a research answer of Experiment04

The research question can be answered as follows:
If if a process changes leads to a different product type distribution (in this case a sub-set of trained products) 
that does not correspond to the CPS's current specialization,
the CPS's current specialization adapts to the currently required set of skills
and preserves once learnt knowledge effectively while irelevant knowledge is forgotten.
Please remark the different performance levels in terms of `accuracy` and `loss`.
It might be better to learn on a new ANN.

On the other hand, the corresponding data set manipulation shows as adequate mechanism to partly forget knowledge:
The one kind of knowledge shall be preserved, while the other kind of knowledge shall be forgotten.

A publication about this is in progress.