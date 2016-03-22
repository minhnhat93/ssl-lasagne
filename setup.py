#!/usr/bin/env python

import os
import urllib
from subprocess import call, check_call

# Current Path of the file and data folder
dataPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
path = os.path.dirname(os.path.realpath(__file__))

# Get files, want to check if we have mnist data base
files = os.listdir(dataPath)

# Download and unzip file
if 'mnist.pkl' not in files:
    print "Downloading MNIST"

    # Inside Data folder
    os.chdir(dataPath)
    mnist = urllib.URLopener()
    mnist.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
    call(['gunzip', 'mnist.pkl.gz'])


def getInput():
    while True:
        answer = raw_input()
        answer.lower()

        if answer == 'y' or answer == 'yes':
            # Setup env
            return True
        elif answer == 'n' or answer == 'no':
            return False


# In root path now
os.chdir(path)

print "Do you want to setup a virtual enviroment? (y/n) "
useenv = getInput()

# Put env in env
if useenv:
    check_call(['virtualenv', '--system-site-packages', 'env'])
    # Next Install requirments for package
    environment = os.environ.copy()
    environment['PATH'] = os.pathsep.join([os.path.join(path, "env/bin"), environment['PATH']])
    call(['pip', 'install', '-r', 'requirements.txt'], env=environment)
    print
    print "To enter the virtual enviroment use: 'source env/bin/activate'"

else:
    # Install in user folder
    print
    print "Do you want to install the requirments globally? Otherwise at user level (y/n)"
    if getInput():
        call(['pip', 'install', '-r', 'requirements.txt'])
    else:
        call(['pip', 'install', '--user', '-r', 'requirements.txt'])
print
print "Installation complete!"
print "To use GPU with Theano use 'THEANO_FLAGS='device=gpu0' python semisupervised_lasagne.py'"
print "OR see Theano documentation for using .theanorc"
