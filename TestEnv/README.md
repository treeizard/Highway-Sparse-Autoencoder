# 1. Run Different Tests
## 1.1. Perpetuate Specific Neurons
- `run_dynamic_test.py`: Testing the effect of changing the value of a given neuron. 

## 1.2. Metamorphic Testing
### 1.2.1. Metamorphic Relations:
- Changing the position of the vehicle in queue should not affect the vehicle's decision in a given Scenario.
### 1.2.2. Testing Files
- `run_metamorphic_test_dynamic.py`: 
    - **Change**: switch 2-3, switch 2-4, switch 3-4. 
    - **Examine**: different features and use that to match the similar neurons.
    - Define each scene based on 5 steps. Identify the scenes with similar response in different test cases: Hence we can then define the
    
