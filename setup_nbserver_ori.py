#!/usr/bin/env python
import os

if os.path.isdir("$HOME/.ipython/profile_nbserver")is False:
    os.system("ipython profile create nbserver")
else: 
    os.system("echo profile_nbserver is already exist.")


    
yes = set(['yes','y', 'ye', ''])
no = set(['no','n'])

print ("Do you want to reset password? (y/n)")
choice = raw_input().lower()
if choice in yes:
    from IPython.lib import passwd
    pwsha = passwd()
   
    config_str = """
    # Server config
    c = get_config()
    c.NotebookApp.ip = '*'
    c.NotebookApp.open_browser = False
    c.NotebookApp.password = u'{}'
    # It is a good idea to put it on a known, fixed port
    c.NotebookApp.port = 8888
    c.NotebookApp.notebookr_dir = u'/home/biadmin/'
    """.format(pwsha)
#elif choice in no:
#else:
   #print("Please respond with 'y' or 'n'")

config_str = """
# Server config
c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
# It is a good idea to put it on a known, fixed port
c.NotebookApp.port = 8888
c.NotebookApp.notebookr_dir = u'/home/biadmin/'
"""

home_dir = os.environ['HOME']

with open(home_dir+"/.ipython/profile_nbserver/ipython_notebook_config.py", "w") as cf:
	cf.write(config_str)


os.system("screen -dRR -dmS ipython_notebook ipython notebook --profile=nbserver;")

# https://www.gnu.org/software/screen/manual/screen.html
# screen install check
# 