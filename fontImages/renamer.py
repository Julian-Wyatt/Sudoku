import os
for i in range (1,10):
    files = os.listdir("./"+str(i)+"/")

    for j in range(len(files)):
        if files[j][0]!="G":
            os.rename("./"+str(i)+"/"+files[j],"./"+str(i)+"/TESTING-"+str(j+1)+".jpg")
    files = os.listdir("./"+str(i)+"/")
    for j in range(len(files)):
        if files[j][0] != "G":
            os.rename("./"+str(i)+"/"+files[j],"./"+str(i)+"/"+str(i)+"-"+str(j+1)+".jpg")