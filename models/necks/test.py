import numpy as np
#feature=h*w*c, kernal=k*k*c*c'
def padding(feature,k,mode=“zero”):#H*W*C
    padder={}
    feature=np.array(feature)
    for i in range(feature[2]):
        H,W=feature[0],feature[1]
        h2=h1=np.zeros(k,W+2*k)
        w2=w1=np.zeros(H,k)
        padding=np.hstack(np.hstack(w1,feature[0:2]),w2)
        padding=np.vstack(np.vstack(h1,padding),v2)
        padder.append(padding.tolist)#C*(H+k)*(W+k)
    padder=np.array(padder).transpose(2,0,1)
    return padder

def flatten_f(feature,k):
    array=np.flatten(feature[0,0,0])
        for h in range (1:feature[0]-k):
            for w in range(1:feature[1]-k):
                array=np.vstack(array,np.flatten(feature[h:h+k+1,w:W+k+1;0]))
    for c in range(1:feature[2]):
        arr=np.flatten(feature[0,0,c])
        for h in range (1:feature[0]-k):
            for w in range(1:feature[1]-k):
                arr=np.vstack(arr,np.flatten(feature[h:h+k+1,w:W+k+1;c]))
        array=np.hstack(array,arr)
    return array

def flatten_k(kernal):# k*k*Cin*Cout
    array=np.flatten(kernal[:,:,0,0])
    for cin in range (1:kernal[2]):
        array=np.vstack(array,np.flatten(kernal[:,:,cin,0]).transpose(1,0))
    
    for cout in range(1:kernal[3]):
        arr=np.flatten(kernal[:,:,0,cout])
        for cin in range (1:kernal[2]):
            arr=np.vstack(arr,np.flatten(kernal[:,:,cin,cout]).transpose(1,0))
        array=np.hstack(array,arr)
    return array

def img2col(feature,kernal):
    feature=padding(feature)
    return flatten_f(feature)*flatten_k(kernal)
