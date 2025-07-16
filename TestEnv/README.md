# 1. Run Different Tests
## 1.1. Perpetuate Specific Neurons
- `run_dynamic_test.py`: Testing the effect of changing the value of a given neuron. 

## 1.2. Metamorphic Testing
### 1.2.1. Metamorphic Relations:
- Changing the position of the vehicle in queue should not affect the vehicle's decision in a given Scenario.
### 1.2.2. Testing Files
- `run_metamorphic_test_dynamic.py`: 
    - **Change**: switch 2-3, switch 2-4, switch 3-4. 

    | Original Scenario | Transformed Scnario    | MT | Follow-up |
    |------------|-----------|------------|-----------|
    | Survived    | Survived  | Success       | Q1 |
    | Survived      | Failed  | Fail       | Q2 |
    | Failed | Survived  | Fail       | Q2 |
    | Failed    | Failed       | Success       | Q1 |

- In **traditional MT**, once we received the test result, we are unable to conduct further analysis on the system. Of cause, we can create more complex tests and follow up tests, however the current test case we have generated becomes not useful. Example questions we may want to ask includes:

**Question 1: Neuron Equivalence**

- Which neurons are shared by two models to ensure the similar behaviour? 
    - Compare the Neuron Activation space for these scenes, identify the top $k_2$ most similar neurons. 
    - Deactivate/Activate the differentiated neurons and observe their effect at the point of deviation.
     - Test each neuron if deactivation caused the model to change policy then mark these neurons as the `function neurons` for a given `scenario`. 

**Question 2:**

- Did the model utilize a differnt survival strategy to the original one? 
    - For each variable, isolate the scenario with similar actions, identify the points of deviation.
        - Where position deviates?
        - Where the kinematics deviates? 
    - Compare the Neuron Activation space for these scenes, identify the top $k_2$ most different neurons. 
    - Deactivate/Activate the differentiated neurons and observe their effect at the point of deviation.
    - Test each neuron if deactivation caused the model to move back its original policy then these neurons can be marked as the `policy neurons` for a given `scenario`. 



- **Examine**: different features and use that to match the similar neurons.
    - Define each scene by dividing a scenario based on 5 steps. Identify the scenes with similar response under different test cases. 
    
