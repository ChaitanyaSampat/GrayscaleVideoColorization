import glob
 
files = glob.glob('./infut/*.mp4')
print(type(files))
for file in files:
    print(type(file))
    print(file)