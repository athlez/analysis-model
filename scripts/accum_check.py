"""Accumulate small motion candidates over a clip to reveal a ball path.
Efficient cv2-only (no full ingest). Colours candidates by time."""
import argparse, cv2, numpy as np, os
ap=argparse.ArgumentParser(); ap.add_argument("--video",required=True)
ap.add_argument("--out",required=True); ap.add_argument("--lo",type=int,default=0)
ap.add_argument("--hi",type=int,default=0); ap.add_argument("--scale",type=float,default=0.25)
a=ap.parse_args()
cap=cv2.VideoCapture(a.video); n=int(cap.get(7))
hi=a.hi if a.hi>0 else n
# median bg from sparse samples
samp=[]; fi=0
while True:
    ok,fr=cap.read()
    if not ok: break
    if fi%8==0: samp.append(cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY))
    fi+=1
cap.release()
bg=np.median(np.stack(samp),axis=0).astype(np.uint8)
cap=cv2.VideoCapture(a.video); fi=0; base=None; pts=0
while True:
    ok,fr=cap.read()
    if not ok: break
    if a.lo<=fi<=hi:
        if base is None or fi==(a.lo+hi)//2: base=fr.copy()
        g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
        d=cv2.absdiff(g,bg); _,m=cv2.threshold(d,20,255,cv2.THRESH_BINARY)
        m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        k=(fi-a.lo)/max(hi-a.lo,1); col=(int(255*k),0,int(255*(1-k)))
        for c in cnts:
            ar=cv2.contourArea(c); x,y,w,h=cv2.boundingRect(c)
            if 4<=ar<=1500 and max(w,h)<=100:
                if base is not None: cv2.circle(base,(int(x+w/2),int(y+h/2)),6,col,-1); pts+=1
    fi+=1
cap.release()
if base is not None:
    small=cv2.resize(base,(int(base.shape[1]*a.scale),int(base.shape[0]*a.scale)))
    cv2.imwrite(a.out,small)
print(f"done: {pts} candidate points, frames {a.lo}-{hi}, saved {a.out}")
