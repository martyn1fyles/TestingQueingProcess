import numpy as np
import pandas as pd
import numpy.random as npr

# just to check that testing is working
def increment(x):
    return x + 1

class queueing_process():

    def __init__(self,
        days_to_simulate: int,
        capacity: list,
        demand: list ,
        symptom_onset_to_joining_queue_dist,
        test_processing_delay_dist):

        self.days_to_simulate = days_to_simulate
        self.capacity = capacity
        self.demand = demand
        self.symptom_onset_to_joining_queue_dist = symptom_onset_to_joining_queue_dist
        self.test_processing_delay_dist = test_processing_delay_dist

        # model variables
        self.time = 0
        self.total_applicants = sum(demand)
        self.todays_leavers = None

        self.preallocate_applicant_rows()
        self.preallocate_queue_rows()

    def preallocate_queue_rows(self):
        """We preallocate all the rows of the dataframe for speed of computation. This function specifically creates the dataframe that provides
        an overview of the queue.
        """

        # create a dataframe to store information about the overall queueing process
        self.queue_info = pd.DataFrame({
            'time': list(range(self.days_to_simulate)),
            'new_applicants': self.demand,
            'capacity': self.capacity,
        })

        # create some empty columns for storing results
        self.queue_info['spillover_to_next_day'] = ''
        self.queue_info['total_applications_today'] = ''
        self.queue_info['capacity_exceeded'] = ''
        self.queue_info['capacity_exceeded_by'] = ''
        self.queue_info['number_swabbed_today'] = ''
        self.queue_info['number_left_queue_not_tested'] = ''

    def preallocate_applicant_rows(self):
        """We preallocate all the rows of the dataframe for speed of computation. The function creates a dataframe containing
        individual level information.
        """

        # the list of times that the system will step through 
        timepoints = list(range(self.days_to_simulate + 1))

        # We work of the list of times that people will join the 
        # for example, if 3 people join on day 0, then the list should look like [0,0,0,1,...]
        queue_joining_times = [item for item, count in zip(timepoints, self.demand) for i in range(count)]

        # work out how long each individual waited before joining the queue
        symptom_onset_to_joining_queue_delays = [
            self.symptom_onset_to_joining_queue_dist()
            for applicant in range(self.total_applicants)
        ]

        # subtract the symptom onset to joining queue delay 
        symptom_onsets = np.array(queue_joining_times) - np.array(symptom_onset_to_joining_queue_delays)

        # queue leaving times:
        #TODO: make this a random variable or something
        # currently we just do 7 days since symptom onset
        queue_leaving_times = np.array(symptom_onsets) + 7

        data_dict = {
            'symptom_onset': symptom_onsets,
            'time_entered_queue': queue_joining_times,
            'time_will_leave_queue': queue_leaving_times
        }

        # create the dataframe and columns with empty value for the rest of the datapoints, that are currently unknown
        self.applicant_info = pd.DataFrame(data_dict)
        self.applicant_info['swabbed'] = False
        self.applicant_info['time_swabbed'] = ''
        self.applicant_info['time_received_result'] = ''
        self.applicant_info['waiting_to_be_swabbed'] = False # default value, initially the queue is empty
        self.applicant_info['left_queue_not_swabbed'] = ''

    def update_new_joiners(self):
        """These individuals joined the queue today, and are now waiting to be swabbed
        """

        # These people joined the queue today
        new_applicants = self.applicant_info.time_entered_queue == self.time

        # Set their waiting to be swabbed status to True
        self.applicant_info.loc[new_applicants, 'waiting_to_be_swabbed'] = True
        
    def update_queue_leaver_status(self):
        """These individuals have been in the queue too long. They are no longer trying/able to get a swab, we update their status'
        to reflect this.
        """

        # These people will leave the queue today
        self.leavers = (self.applicant_info.time_will_leave_queue <= self.time) & (self.applicant_info.waiting_to_be_swabbed == True)

        # record the number of people who carry over to the next day
        if self.todays_capacity > len(self.todays_applicants):

            spillover_to_next_day = 0

        else:
            # todays demand - minus those who were served (capacity) - those who leave the queue not being served
            spillover_to_next_day = len(self.todays_applicants) - sum(self.leavers) - self.todays_capacity
        
        #
        self.queue_info.loc[self.time, ['spillover_to_next_day', 'number_left_queue_not_tested']] = [spillover_to_next_day, sum(self.leavers)]

        # Set their waiting to be swabbed status to False
        self.applicant_info.loc[self.leavers, ['waiting_to_be_swabbed', 'left_queue_not_swabbed']] = [False, True]

    @property
    def todays_applicants(self):
        return list(self.applicant_info[self.applicant_info.waiting_to_be_swabbed].index)

    @property
    def todays_capacity(self):
        return int(self.queue_info[self.queue_info.time == self.time].capacity)

    def process_day_of_queue(self):
        """Performs swabbing of individuals.
        """
        
        number_applicants_today = len(self.todays_applicants)

        self.queue_info.loc[self.queue_info.time == self.time, ['total_applications_today']] = [number_applicants_today]

        # is todays capacity exceeded?
        if number_applicants_today < self.todays_capacity:

            # Then the capacity was not exceeded
            self.swab_applicants(to_be_swabbed = self.todays_applicants)

            # record that the capacity was not exceed
            self.queue_info.loc[self.queue_info.time == self.time, ['capacity_exceeded', 'capacity_exceeded_by']] = [False, 0]

        else:
            # then the capacity was exceeded, and only some of todays applicants will be processed
            capacity_exceeded_by = number_applicants_today - self.todays_capacity

            # we assume that of todays applicants, the successful applicants are chosen uniformly at random
            # TODO: add models of how to vary this
            successful_applicants = npr.choice(a = self.todays_applicants, size = self.todays_capacity, replace = False)
            self.swab_applicants(to_be_swabbed = successful_applicants)

            # record that the capacity was exceeded
            self.queue_info.loc[self.queue_info.time == self.time, ['capacity_exceeded', 'capacity_exceeded_by']] = [True, capacity_exceeded_by]

    def swab_applicants(self, to_be_swabbed: list):
        """For a list of applicants who were successful in getting thorugh the queue, update their variables associated with swabbing

        Args:
            to_be_swabbed (list): A list of integers, referring the rows of the applicant_dataframe that will get processed
        """
        
        # The columns that will be updated
        columns_to_update = [
            'waiting_to_be_swabbed',
            'time_swabbed',
            'left_queue_not_swabbed',
            'swabbed'
        ]

        # record an attribtue of which individuals were swabbed today for use later
        self.todays_swabbed_index = to_be_swabbed

        # update the above status to show they have been swabbed
        self.applicant_info.loc[to_be_swabbed, columns_to_update] = [False, self.time, False, True]

        # how long does it take to process their test?
        test_result_processing_times = np.array([self.test_processing_delay_dist() for swabbed_individual in range(len(to_be_swabbed))])

        # work out when they receive their result, and update the data
        self.applicant_info.loc[to_be_swabbed, 'time_received_result'] = self.time + test_result_processing_times

        # update the queue_info table with the number of individuals processed today
        self.queue_info.loc[self.queue_info.time == self.time, ['number_swabbed_today']] = len(to_be_swabbed)

    def simulate_one_day(self, verbose = True):
        """Simulates one day of the queue.

        Args:
            verbose (bool, optional): If true, output simulation progress. Defaults to True.
        """

        # steps required to simulate one day
        self.update_new_joiners()
        self.update_queue_leaver_status()
        self.process_day_of_queue()

        # update the model time
        self.time += 1

        # make a nice little status update
        if verbose:
            print(f'Model time {self.time}, progress: {round((self.time + 1) / self.days_to_simulate * 100)}%', end = '\r')

    def run_simulation(self, verbose = True):
        """Runs the queueing process model.
        """

        while self.time < self.days_to_simulate:

            self.simulate_one_day(verbose = verbose)

    
    def get_delays_for(self, time_entered_queue: int, delay_from_column: str, delay_to_column: str):
        """Return a list of the delays between two timepoints who joined on a specified day

        Args:
            time (int): The day on which the applicants joined the queue
            delay_from_column (str): The earliest timepoint
            delay_to_column (str): The latest timepoint
        """
        day_index = (self.applicant_info.time_entered_queue == time_entered_queue) & (self.applicant_info.swabbed == True)
        delay_from_column = self.applicant_info.loc[day_index, delay_from_column]
        delay_to_column = self.applicant_info.loc[day_index, delay_to_column]
        return delay_to_column - delay_from_column
    
    def get_prob_getting_tested(self, time_entered_queue: int):
        """Returns the probability of getting tested if you join the queue on a specified day

        Args:
            time_entered_queue (int): The day of interest
        """
        valid_individuals = (self.applicant_info.time_entered_queue == time_entered_queue) & (self.applicant_info.waiting_to_be_swabbed == False)
        left_queue_not_swabbed = self.applicant_info[valid_individuals].left_queue_not_swabbed
        return 1 - left_queue_not_swabbed.mean()

class dynamic_queueing_process(queueing_process):

    def __init__(self,
        test_processing_delay_dist,
        days_to_simulate: int,
        capacity: list):
        """A dynamic queueing process where individuals can be added to the queue at the beginning of a day. In the
        static queueing model, demand is determined prior to running the queue

        Args:
            test_processing_delay_dist (func): A function that returns an integer value representing the number of days it takes to process a test
            days_to_simulate (int): The number of time steps the model will step through
            capacity (list): A list containing the capacity of the system over time
        """

        self.test_processing_delay_dist = test_processing_delay_dist
        self.days_to_simulate = days_to_simulate
        self.capacity = capacity

        # create the dataframe and columns with empty value for the rest of the datapoints, that are currently unknown and will be dynamically generated by the branching process
        self.applicant_info = pd.DataFrame()
        self.applicant_info['node_id'] = ''
        self.applicant_info['swabbed'] = False
        self.applicant_info['time_joined_queue'] = ''
        self.applicant_info['time_swabbed'] = ''
        self.applicant_info['time_received_result'] = ''
        self.applicant_info['waiting_to_be_swabbed'] = False # default value, initially the queue is empty
        self.applicant_info['left_queue_not_swabbed'] = ''

        # create the dataframe that records queue statistics
        self.preallocate_queue_rows()

    def preallocate_queue_rows(self):
        """We preallocate all the rows of the dataframe for speed of computation. This function specifically creates the dataframe that provides
        an overview of the queue.
        """

        # create a dataframe to store information about the overall queueing process
        self.queue_info = pd.DataFrame({
            'time': list(range(self.days_to_simulate)),
            'capacity': self.capacity,
        })

        # create some empty columns for storing results
        self.queue_info['new_applicants'] = ''
        self.queue_info['spillover_to_next_day'] = ''
        self.queue_info['total_applications_today'] = ''
        self.queue_info['capacity_exceeded'] = ''
        self.queue_info['capacity_exceeded_by'] = ''
        self.queue_info['number_swabbed_today'] = ''
        self.queue_info['number_left_queue_not_tested'] = ''

    def add_new_queue_joiners(self,
        new_joiner_node_ids: list,
        new_joiner_symptom_onsets: list):
        """Add individuals who are now attempting to get tested to the queue for processing

        Args:
            new_joiner_node_ids (list): [description]
            new_joiner_symptom_onsets (list): [description]
        """

        # queue leaving times:
        #TODO: make this a random variable or something
        # currently we just do 7 days since symptom onset
        queue_leaving_times = np.array(new_joiner_symptom_onsets) + 7

        data_dict = {
            'node_id': new_joiner_node_ids,
            'symptom_onset': new_joiner_symptom_onsets,
            'time_will_leave_queue': queue_leaving_times
        }

        # create the dataframe and columns with empty value for the rest of the datapoints, that are currently unknown
        new_applicants_info = pd.DataFrame(data_dict)
        new_applicants_info['time_joined_queue'] = self.time
        new_applicants_info['swabbed'] = False
        new_applicants_info['time_swabbed'] = ''
        new_applicants_info['time_received_result'] = ''
        new_applicants_info['waiting_to_be_swabbed'] = False # default value, initially the queue is empty
        new_applicants_info['left_queue_not_swabbed'] = ''

        self.applicant_info.append(new_applicants_info, ignore_index = True)

    def get_todays_queue_output(self):

        swabbed_individuals = self.applicant_info.loc[self.todays_swabbed_index]

        output = {
            'leaving_the_queue_node_ids': self.todays_leavers,
            'swabbed_individuals': swabbed_individuals
        }
        
        return output
