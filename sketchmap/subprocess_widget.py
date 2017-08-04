from os import path
import re
import time
import threading
import subprocess
import ipywidgets as widgets

class SubprocessWidget(widgets.Box):
    def __init__(self, cmd, log_fn):
        self.cmd = cmd
        self.log_fn = log_fn
        self.process = None
        self.btn_startstop = widgets.Button(description="Start")
        self.output_area = widgets.HTML()
        self.btn_startstop.on_click(self.on_click_startstop)
        super(SubprocessWidget, self).__init__([self.btn_startstop, self.output_area])
        
    def output_worker(self):
        pre_tag = '<pre style="width:600px; max-height:300px; overflow-x:auto; line-height:1em; font-size:0.8em;">'
        dots = 0
        latest = ""
        while(self.process.poll() == None):
            time.sleep(1)
            # update output window
            if path.exists(self.log_fn):
                full = open(self.log_fn).read() # TODO seek forward
                full = re.sub("\n.*\r", "\n", full.strip("\r")) # handle carriage return
                output = "\n".join(full.split("\n")[-20:]) #last 100 lines
            else:
                output = ""
            dots += 1
            if(latest!=output): # new output
                dots = 0 # rest dot counter
            latest = output
            output += "\n" + ("."*dots) + "\n"
            self.output_area.value = pre_tag + output + '</pre>'

        # read one last time entirely
        output = open(self.log_fn).read()
        output = re.sub("\n.*\r", "\n", output.strip("\r")) # handle carriage return
        output += "\n\nProcess finished, exit code: %s"%self.process.returncode
        self.output_area.value = pre_tag + output + '</pre>'
        self.btn_startstop.description="Start"
    
    def start_process(self):
        self.btn_startstop.description="Stop"
        logfile = open(self.log_fn, "w")
        self.process = subprocess.Popen(self.cmd, stdout=logfile, stderr=logfile)
        threading.Thread(target=self.output_worker).start() 

    def on_click_startstop(self, e):
        if(self.btn_startstop.description.startswith("Start")):
            self.start_process()
        else:
            self.process.kill()
#EOF