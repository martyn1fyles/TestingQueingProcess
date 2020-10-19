import queueing_process as qp
import numpy as np

# a chunk of code that can be used to run the model in a way that is suitable for debugging

def symptom_onset_to_joining_queue():
    return 2

def test_processing_delay_dist():
    return 1

days_to_simulate = 50
demand = [int(round(500*np.exp(0.05 * time))) for time in range(days_to_simulate)] 
capacity = [500 + time * 10 for time in range(days_to_simulate)]

my_queue = qp.queueing_process(
    days_to_simulate = days_to_simulate,
    capacity = capacity,
    demand = demand,
    symptom_onset_to_joining_queue_dist = symptom_onset_to_joining_queue,
    test_processing_delay_dist = test_processing_delay_dist
)

my_queue.run_simulation()
