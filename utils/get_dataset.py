import requests
import tarfile

def prepare_data():
    print("Preparing data")
    print("Downloading corpus...")
    with requests.get('https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/nfcorpus.tar.gz' , 
        stream=True, auth=('user', 'pass')) as  rx,\
        tarfile.open(fileobj=rx.raw  , mode="r:gz") as tarobj  :
        print("Extracting files...")
        tarobj.extractall() 
   
    print("All files downloaded...")

if __name__ == '__main__':
    prepare_data()