import queueing_process.queueing_process as qp

# example test

def test_increment():
    assert qp.increment(3) == 4

def simple_onset_to_queue_dist():
    return 1 

def simple_test_processing_delay_dist():
    return 1

def test_preallocate_queue_rows():
    """Tests the function which creates the pandas dataframe
    """
    test_queue = qp.queueing_process(
        days_to_simulate = 3,
        capacity = [2, 3, 4],
        demand = [3, 4, 5],
        symptom_onset_to_joining_queue_dist = simple_onset_to_queue_dist,
        test_processing_delay_dist = simple_test_processing_delay_dist)

    test_queue.queue_info
