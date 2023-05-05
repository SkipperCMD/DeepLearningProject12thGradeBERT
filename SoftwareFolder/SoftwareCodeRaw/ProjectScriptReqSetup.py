import os
import sys
import subprocess

ReqPath = os.path.dirname(__file__) + "\ApplicationResources\ProjectRequirementsPackages.txt"
print("Now installing Python Packages....")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', ReqPath])
print("Completed!")