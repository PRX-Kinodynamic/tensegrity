uint8 PREPARING=0 #  Initial status: The node is "up" but not ready to operate (i.e. is reading a config file)
uint8 READY=1     #  The node is either ready to run or actively running
uint8 RUNNING=2   #  The node is either ready to run or actively running
uint8 WAITING=3   #  The node was "running" but it is no longer running, i.e. paused. It may go back to running
uint8 STOPPED=4   #  The node has been stopped and will exit soon. May be doing some cleanup (i.e. writing to file).
uint8 FINISH=5   #  The node has been stopped and will exit soon. May be doing some cleanup (i.e. writing to file).
uint8 ERROR=6   #  The node has been stopped and will exit soon. May be doing some cleanup (i.e. writing to file).

uint8 status
