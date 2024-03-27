import os
 
dir = './tempo/'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))