import glob
import cv2
import natsort
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
from skimage.measure import compare_ssim

#img_path1=natsort.natsorted(glob.glob('amodal2/10-29-2020/mask/*.png'),reverse=False)
#img_path3=natsort.natsorted(glob.glob('amodal2/10-29-2020/images/*.jpg'),reverse=False)

img_path2=natsort.natsorted(glob.glob('outdir5/*.png'),reverse=False)
alpha=0.5
from sklearn.metrics import confusion_matrix
import numpy as np
def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def compute_bottom_iou(y_pred, y_true,res):
    # ytrue, ypred is a flatten vecto
    exc2=[128, 64, 128]
    exc1=[0, 0, 0]
    #print(image.shape)
    indices_list=[]
    indices_list2=[]
    ct=0
    ctt=0
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            #print(y_pred[i,j])
            if np.all(y_pred[i,j]==[255,255,255]) or np.all(y_true[i,j]==[1,1,1]) and res[i,j]==0:
                ct=ct+1
                if np.all(y_pred[i,j]==[255,255,255]) and np.all(y_true[i,j]==[1,1,1]) and res[i,j]==0:
                    ctt=ctt+1
    return (ctt,ct)

def find_far(depth):
    # ytrue, ypred is a flatten vecto
    tar=0
    flag=False
    res=np.zeros((256,512))
    for i in reversed(range(len(depth))):
        #print(np.max(depth[i]))
        if depth[i][depth[i]!=0].mean()>30 and flag==False:
            tar=i
            flag=True
            #print(np.max(depth[i]))
        for j in range(len(depth[i])):
            if depth[i,j]>30:
                res[i,j]=1

    return tar,res

def compute_top_iou(y_pred, y_true,res):
    # ytrue, ypred is a flatten vecto
    exc2=[128, 64, 128]
    exc1=[0, 0, 0]
    #print(image.shape)
    indices_list=[]
    indices_list2=[]
    ct=0
    ctt=0
    for i in range(len(y_pred)):
        for j in range(len(y_pred[i])):
            #print(y_pred[i,j])
            if np.all(y_pred[i,j]==[255,255,255]) or np.all(y_true[i,j]==[1,1,1]) and res[i,j]==1:
                ct=ct+1
                if np.all(y_pred[i,j]==[255,255,255]) and np.all(y_true[i,j]==[1,1,1]) and res[i,j]==1:
                    ctt=ctt+1
    return (ctt,ct)


def foregroud_iou(y_pred,y_true,fore):
    exc2=[128, 64, 128]
    exc1=[0, 0, 0]
    #print(image.shape)
    ct=0
    ctt=0
    for i in range(len(y_pred)):
        for j in range(len(y_pred)):
            #print(y_pred[i,j])
            if np.all(fore[i,j]==[255,255,255]):
                ct=ct+1
                if np.all(y_pred[i,j]==y_true[i,j]):
                    ctt=ctt+1

    return (ctt,ct)

metric=0
em=0
ff=0
res=[]
de_c=0
de_f=0
n_c=0
n_f=0
with open("collegetown_depth2.csv") as f:
    lis = [line.split() for line in f]
    for j in range(len(lis)):
        img_path1=lis[j][0].split(",")[-2]
        print(lis[j][0].split(",")[-1])
        depth_img=cv2.resize(np.load(lis[j][0].split(",")[-1]),(512,256))
        print(img_path1)
        img_path3=lis[j][0].split(",")[0]
        fore_path=lis[j][0].split(",")[1]
        #img_path2=natsort.natsorted(glob.glob('test13/*.png'),reverse=False)
        img=cv2.imread(img_path3)
        im=cv2.imread(img_path2[j])
        im2=cv2.resize(img,(512,256))
        mask=cv2.imread(img_path1)
        mask=cv2.resize(mask,(512,256))
        overlay = im2.copy()
        output = im2.copy()
        #exc1=[244, 35, 232]

        tar,res=find_far(depth_img)
        print(tar)
        im_bottom = im[tar: 256, 0:512]
        mask_bottom = mask[tar: 256, 0:512]
        #cv2.imwrite("test.png",mask_bottom)
        (tmp21,tmp22)=compute_bottom_iou(im_bottom,mask_bottom,res)
        n_c=n_c+tmp21
        de_c=de_c+tmp22
        print(n_c/de_c)
        im_top = im[0: tar, 0:512]
        mask_top = mask[0: tar, 0:512]
        #cv2.imwrite("test.png",im2[200: 256, 0:512])
        (tmp31,tmp32)=compute_top_iou(im_top,mask_top,res)
        n_f=n_f+tmp31
        de_f=de_f+tmp32
        #print(n_f/de_f)
        exc2=[255,255,255]
        #print(image.shape)
        #indices_list=np.where(np.any(img==exc1,axis=-1))
        indices_list2=np.where(np.any(im==exc2,axis=-1))
        #im2[indices_list]=(244, 35, 232)
        overlay[indices_list2]=(203, 192, 255)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,
        		0, output)
        output1=cv2.line(output, (0,tar), (512,tar), (0, 255, 0), thickness=2)
        output1=cv2.putText(output1,  str("{:.4f}".format(n_c/de_c)), (400,250), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 2, cv2.LINE_AA)
        output1=cv2.putText(output1,  str("{:.4f}".format(n_f/de_f)), (400,tar), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite("over_vad/test "+str(j)+".png",output1)




        '''
        c=np.array([0,0,0])
        #print((mask==b).all(axis=2))
        indices_list=np.where(np.all(mask==c,axis=-1))
        indices_list2=np.where(np.any(mask!=c,axis=-1))
        #mask2=mask
        #b = np.array([255,255,255])
        mask[indices_list]=255
        mask[indices_list2]=0
        #cv2.imwrite("test.png",mask)
        tmp=compute_edge(cv2.Canny(im,100,200),cv2.Canny(mask,100,200))
        em=em+tmp
        print(tmp)

        fore_img=cv2.imread(fore_path)
        fore_img=cv2.resize(fore_img,(512,256))
        c=np.array([0,0,0])
        #print((mask==b).all(axis=2))
        indices_list2=np.where(np.any(fore_img!=c,axis=-1))
        fore=fore_img
        #b = np.array([255,255,255])
        fore[indices_list2]=[255,255,255]
        tmp3=foregroud_iou(im,mask,fore)
        ff=ff+tmp3
        print(tmp3)
        res.append(tmp2)
        '''
'''
import matplotlib.pyplot as plt
plt.hist(np.array(res), bins=np.linspace(0.5,1,10), ec='black')
plt.show()
'''
'''
res.sort(key=lambda x: x[0])
final_path=[i[1] for i in res[:300] if i[0]<0.9]
final_sum=[i[0] for i in res[:300] if i[0] <0.9]
import pandas as pd
import csv
path=[]
test=pd.read_csv("test15.csv")
for i, row in test.iterrows():
    path1,path2,path3=row
    path.append(path1)
with open('test15.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    for j in final_path:
        if j not in path:
            writer.writerow([j,"/media/bizon/Elements/amodal_dataset3/"+j.split("/")[-3]+"/seg/"+j.split("/")[-1],"/media/bizon/Elements/amodal_dataset3/"+j.split("/")[-3]+"/mask/"+j.split("/")[-1]])
'''
print(n_c/de_c)
print(n_f/de_f)

#print(ff)
