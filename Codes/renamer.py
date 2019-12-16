import os 
  
# Function to rename multiple files 
def main(): 
    i = '0001'; 
    dirc =  "missing\\annots" ;   
    print(dirc)
    for filename in os.listdir(dirc):
        dirc = "missing\\annots\\"
        dst ="mh" + i + ".xml"
        src = dirc + filename 
        dst = dirc + dst 
        print(filename[2:-4])
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i = str(int(i) + 1).zfill(4)
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 