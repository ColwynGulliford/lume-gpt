from time import sleep, time
import sys


def fake_gpt():

    tstart = time()
    while True:

        current_time = time()

        print('Running happily: '+str(current_time-tstart), file=sys.stdout)
        sys.stdout.flush()

        sleep(0.1)

def fake_gpt_hangs():

    hang_after = 5

    tstart = time()

    while True:

        current_time = time()

        if(current_time-tstart < hang_after):
            print('Running happily: '+str(current_time-tstart), file=sys.stdout)
            sys.stdout.flush()
        else:
            print('Stuck :(')
            sys.stdout.flush()
            while True:
                sleep(0.1)

        sleep(0.25)

def fake_gpt_ends():

    end_after = 5

    tstart = time()

    while True:

        current_time = time()

        if(current_time-tstart < end_after):
            print('Running happily: '+str(current_time-tstart), file=sys.stdout)
            sys.stdout.flush()
        else:
            print('Done :)')
            sys.stdout.flush()
            break

        sleep(0.1)

def fake_gpt_errs():

    hang_after = 5

    tstart = time()

    while True:

        current_time = time()

        if(current_time-tstart < hang_after):
            print('Running happily: '+str(current_time-tstart), file=sys.stdout)
            sys.stdout.flush()
        else:
            print('Error!')
            sys.stdout.flush()

        sleep(0.25)



if __name__ == '__main__':

    #fake_gpt_hangs()
    #fake_gpt_ends()
    fake_gpt_errs()

    