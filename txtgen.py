import os

root = "CK+48"
with open('train.txt', 'w') as file:
    pass
with open('val.txt', 'w') as file:
    pass
with open('test.txt', 'w') as file:
    pass

D = {'surprise':0,'sadness':1,'happy':2,'fear':3,'disgust':4,'contempt':5,'anger':6}

def main():
    
    for dir in os.listdir(root):
        fdp = os.path.join(root,dir) 
        label = D[dir]
        for idx,i in enumerate(os.listdir(fdp)):
            fp = os.path.join(fdp,i)
            if idx % 5 == 0:
                with open('val.txt', 'a') as file:
                    file.write(fp+' '+str(label)+'\n')
            elif idx % 5 == 1:
                with open('test.txt', 'a') as file:
                    file.write(fp+' '+str(label)+'\n')
            else:
                with open('train.txt', 'a') as file:
                    file.write(fp+' '+str(label)+'\n')
if __name__ == "__main__":
    main()
    