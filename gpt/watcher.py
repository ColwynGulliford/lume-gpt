import subprocess
import threading
import time 
import sys

class Watcher():
    """
    Watcher class for line by line watching of subprocess with a timeout
    """
    def __init__(self, cmd, verbose=False, timeout=10, kill_msgs=[]):

        """
        cmd: str, command to run via subprocess 
        timeout: float [sec], watcher kills subprocess when t > timeout
        kill_msgs: list(str), list of strings to check from in output lines from subprocess.  
                   If found in output line, subprocess is killed.
        """

        if(isinstance(cmd, str)):
            cmd=cmd.split(' ')

        self.cmd_popen = None
        self.cmd = cmd
        self.verbose=verbose
        self.timeout = timeout
        self.t_event = threading.Event()
        self.t_event.clear()
        self.thread = None
        self.kill_msgs=kill_msgs
        self.timeout_occured=False
        self.exception=None
        self.log = []
        self.run_time = None

    def runner(self):

        """ Timeout monitor function targeted by monitor thread """

        ret = self.t_event.wait(self.timeout)
        # ret will be False if a Timeout happens
        if not ret:
            self.timeout_occured=True
            self.cmd_popen.kill()

    def execute(self):

        """ Runs the subprocess command and starts the monitor thread Yields output lines from subprocess."""

        self.thread = threading.Thread(target=self.runner)
        self.cmd_popen = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        self.thread.start()

        for stdout_line in iter(self.cmd_popen.stderr.readline, ""):
            yield stdout_line

    def run(self):

        """ Main function, calls execute and prints output to screen.  Checks output for kill messages. """
        t1 = time.time()
        for line in self.execute():
            self.log.append(line)
            if(self.verbose):
                print(line.strip('\n'))

            for msg in self.kill_msgs:
                if msg in line:
                    self.cmd_popen.kill()
                    self.exception=line
                    break

        self.t_event.set()
        t2 = time.time()
        self.run_time=t2-t1


if __name__ == '__main__':

    kill_msgs = ['Stuck']

    t1 = time.time()
    w = Watcher(cmd='python fake_gpt.py', timeout=10, kill_msgs=['Error'])
    w.run()
    t2 = time.time()
    print('Time Ellapsed:', t2-t1)
