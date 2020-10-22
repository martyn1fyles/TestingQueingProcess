import sys
sys.path.append('..')
print(sys.path)

import queueing_process.queueing_process_2

def increment(x):
    return x + 1

def test_increment():
    assert increment(3) == 4
