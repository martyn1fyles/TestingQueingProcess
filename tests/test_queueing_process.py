import queueing_process.queueing_process as qp
import pandas as pd
import pytest

def simple_onset_to_queue_dist():
    return 1 

def simple_test_processing_delay_dist():
    return 1

@pytest.fixture
def test_queue():
    capacity_in = [2, 3, 4]
    demand_in = [3, 4, 5]

    test_queue = qp.queueing_process(
        days_to_simulate = 3,
        capacity = capacity_in,
        demand = demand_in,
        symptom_onset_to_joining_queue_dist = simple_onset_to_queue_dist,
        test_processing_delay_dist = simple_test_processing_delay_dist)

    return test_queue

@pytest.fixture
def queue_set_up():
    return pd.read_pickle('tests/fixtures/queue_set_up')

def test_preallocate_queue_rows(test_queue, queue_set_up):
    """Tests the function which creates the pandas dataframe for the queue
    """

    return pd.testing.assert_frame_equal(test_queue.queue_info, queue_set_up)

@pytest.fixture
def applicant_set_up():
    return pd.read_pickle('tests/fixtures/applicant_info_set_up')

def test_preallocate_applicant_rows(test_queue, applicant_set_up):
    """Tests the function which creates the pandas dataframe for the applicants
    """
    
    return pd.testing.assert_frame_equal(test_queue.applicant_info, applicant_set_up)

@pytest.fixture
def applicant_info_new_joiner():
    return pd.read_pickle('tests/fixtures/applicant_info_new_joiner')

def test_new_joiners(test_queue, applicant_info_new_joiner):
    """Tests the function which creates the pandas dataframe for the applicants
    """
    test_queue.update_new_joiners()

    return pd.testing.assert_frame_equal(test_queue.applicant_info, applicant_info_new_joiner)

def test_todays_applicants(test_queue):
    """Tests that todays applicants property returns the index of the individuals
    who are joining the queue on time 0. Update joiners must be run to update the fact that they
    are waiting to be swabbed.
    """

    test_queue.update_new_joiners()

    assert test_queue.todays_applicants == [0, 1, 2] 

def test_todays_capacity(test_queue):
    """Asserts the todays capacity method returns the right capacity for today
    """

    assert test_queue.todays_capacity == 2

def test_todays_capacity_time_incremented(test_queue):
    """Call simulate one day, which increments the model time
    Asserts that the todays capacity is now that of t=1]
    """

    test_queue.simulate_one_day()

    assert test_queue.todays_capacity == 3

@pytest.fixture
def update_queue_leaver_status_applicant_info():
    return pd.read_pickle('tests/fixtures/update_queue_leaver_status_applicant_info')

def test_update_queue_leaver_status_applicant_info(update_queue_leaver_status_applicant_info):

    test_queue = qp.queueing_process(
        days_to_simulate = 7,
        capacity = [1]*7,
        demand = [1]*7,
        symptom_onset_to_joining_queue_dist = simple_onset_to_queue_dist,
        test_processing_delay_dist = simple_test_processing_delay_dist)

    test_queue.update_new_joiners()

    test_queue.time = 6

    test_queue.update_queue_leaver_status()

    return pd.testing.assert_frame_equal(test_queue.applicant_info, update_queue_leaver_status_applicant_info)

@pytest.fixture
def update_queue_leaver_status_queue_info():
    return pd.read_pickle('tests/fixtures/update_queue_leaver_status_queue_info')

def test_update_queue_leaver_status_queue_info(update_queue_leaver_status_queue_info):

    test_queue = qp.queueing_process(
        days_to_simulate = 7,
        capacity = [1]*7,
        demand = [1]*7,
        symptom_onset_to_joining_queue_dist = simple_onset_to_queue_dist,
        test_processing_delay_dist = simple_test_processing_delay_dist)

    test_queue.update_new_joiners()

    test_queue.time = 6

    test_queue.update_queue_leaver_status()

    return pd.testing.assert_frame_equal(test_queue.queue_info, update_queue_leaver_status_queue_info)

@pytest.fixture
def swab_applicants():
    return pd.read_pickle('tests/fixtures/swab_applicants')

def test_swab_applicants(test_queue, swab_applicants):

    test_queue.update_new_joiners()

    test_queue.swab_applicants(to_be_swabbed=[0, 2])

    return pd.testing.assert_frame_equal(test_queue.applicant_info, swab_applicants)

def test_swab_applicants_queue_update(test_queue):
    
    test_queue.update_new_joiners()

    test_queue.swab_applicants(to_be_swabbed=[1, 2])

    assert test_queue.queue_info.number_swabbed_today[0] == 2

def test_process_day_of_queue_capacity_less_than_demand(test_queue):

    test_queue.update_new_joiners()

    test_queue.process_day_of_queue()

    assert test_queue.applicant_info.swabbed.sum() == 2

def test_process_day_of_queue_capacity_less_than_demand_queue_info(test_queue):

    test_queue.update_new_joiners()

    test_queue.process_day_of_queue()

    assert (test_queue.queue_info.loc[0, ['capacity_exceeded', 'capacity_exceeded_by']] == [True, 1]).all()

@pytest.fixture
def process_day_of_queue_demand_less_than_capacity_applicant_info():
    return pd.read_pickle('tests/fixtures/process_day_of_queue_demand_less_than_capacity_applicant_info')

def test_process_day_of_queue_demand_less_than_capacity(process_day_of_queue_demand_less_than_capacity_applicant_info):

    capacity_in = [5, 3, 4]
    demand_in = [3, 4, 5]

    test_queue = qp.queueing_process(
        days_to_simulate = 3,
        capacity = capacity_in,
        demand = demand_in,
        symptom_onset_to_joining_queue_dist = simple_onset_to_queue_dist,
        test_processing_delay_dist = simple_test_processing_delay_dist)

    test_queue.update_new_joiners()
    
    test_queue.process_day_of_queue()

    return pd.testing.assert_frame_equal(test_queue.applicant_info, process_day_of_queue_demand_less_than_capacity_applicant_info)

def test_process_day_of_queue_demand_less_than_capacity_queue_info():

    capacity_in = [5, 3, 4]
    demand_in = [3, 4, 5]

    test_queue = qp.queueing_process(
        days_to_simulate = 3,
        capacity = capacity_in,
        demand = demand_in,
        symptom_onset_to_joining_queue_dist = simple_onset_to_queue_dist,
        test_processing_delay_dist = simple_test_processing_delay_dist)

    test_queue.update_new_joiners()
    
    test_queue.process_day_of_queue()

    assert (test_queue.queue_info.loc[0, ['capacity_exceeded', 'capacity_exceeded_by']] == [False, 0]).all()

def test_simulate_one_day_time(test_queue):

    test_queue.simulate_one_day()

    assert test_queue.time == 1

def test_get_delays_for():

    capacity_in = [3, 3, 4]
    demand_in = [3, 4, 5]

    test_queue = qp.queueing_process(
        days_to_simulate = 3,
        capacity = capacity_in,
        demand = demand_in,
        symptom_onset_to_joining_queue_dist = simple_onset_to_queue_dist,
        test_processing_delay_dist = simple_test_processing_delay_dist)

    test_queue.run_simulation()

    assert (test_queue.get_delays_for(0, 'symptom_onset', 'time_received_result') == 2).all()


