#import napari
import numpy as np,pickle,glob,os
import cv2
from scipy.signal import convolve,fftconvolve
from tqdm.notebook import tqdm
import matplotlib.pylab as plt

from tqdm.notebook import tqdm
def get_p99(fl_dapi,resc=4):
    im = read_im(fl_dapi)
    im_ = np.array(im[-1][im.shape[1]//2],dtype=np.float32)
    img = (cv2.blur(im_,(2,2))-cv2.blur(im_,(50,50)))[::resc,::resc]
    p99 = np.percentile(img,99.9)
    p1 = np.percentile(img,1)
    img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
    plt.figure()
    plt.imshow(img,cmap='gray')
    return p99
def resize(im,shape_ = [50,2048,2048]):
    """Given an 3d image <im> this provides a quick way to resize based on nneighbor sampling"""
    z_int = np.round(np.linspace(0,im.shape[0]-1,shape_[0])).astype(int)
    x_int = np.round(np.linspace(0,im.shape[1]-1,shape_[1])).astype(int)
    y_int = np.round(np.linspace(0,im.shape[2]-1,shape_[2])).astype(int)
    return im[z_int][:,x_int][:,:,y_int]

import scipy.ndimage as ndimage
def get_final_cells_cyto(im_polyA,final_cells,icells_keep=None,ires = 4,iresf=10,dist_cutoff=10):
    """Given a 3D im_polyA signal and a segmentation fie final_cells """
    incell = final_cells>0
    med_polyA = np.median(im_polyA[incell])
    med_nonpolyA = np.median(im_polyA[~incell])
    im_ext_cells = im_polyA>(med_polyA+med_nonpolyA)/2


    X = np.array(np.where(im_ext_cells[:,::ires,::ires])).T
    Xcells = np.array(np.where(final_cells[:,::ires,::ires]>0)).T
    from sklearn.neighbors import KDTree

    kdt = KDTree(Xcells[::iresf], leaf_size=30, metric='euclidean')
    icells_neigh = final_cells[:,::ires,::ires][Xcells[::iresf,0],Xcells[::iresf,1],Xcells[::iresf,2]]
    dist,neighs = kdt.query(X, k=1, return_distance=True)
    dist,neighs = np.squeeze(dist),np.squeeze(neighs)

    final_cells_cyto = im_ext_cells[:,::ires,::ires]*0
    if icells_keep is not None:
        keep_cyto = (dist<dist_cutoff)&np.in1d(icells_neigh[neighs],icells_keep)
    else:
        keep_cyto = (dist<dist_cutoff)
    final_cells_cyto[X[keep_cyto,0],X[keep_cyto,1],X[keep_cyto,2]] = icells_neigh[neighs[keep_cyto]]
    final_cells_cyto = resize(final_cells_cyto,im_polyA.shape)
    return final_cells_cyto
def slice_pair_to_info(pair):
    sl1,sl2 = pair
    xm,ym,sx,sy = sl2.start,sl1.start,sl2.stop-sl2.start,sl1.stop-sl1.start
    A = sx*sy
    return [xm,ym,sx,sy,A]
def get_coords(imlab1,infos1,cell1):
    xm,ym,sx,sy,A,icl = infos1[cell1-1]
    return np.array(np.where(imlab1[ym:ym+sy,xm:xm+sx]==icl)).T+[ym,xm]
def cells_to_coords(imlab1,return_labs=False):
    """return the coordinates of cells with some additional info"""
    infos1 = [slice_pair_to_info(pair)+[icell+1] for icell,pair in enumerate(ndimage.find_objects(imlab1))
    if pair is not None]
    cms1 = np.array([np.mean(get_coords(imlab1,infos1,cl+1),0) for cl in range(len(infos1))])
    cms1 = cms1[:,::-1]
    ies = [info[-1] for info in infos1]
    if return_labs:
        return imlab1.copy(),infos1,cms1,ies
    return imlab1.copy(),infos1,cms1
def resplit(cells1,cells2,nmin=100):
    """intermediate function used by standard_segmentation.
    Decide when comparing two planes which cells to split"""
    imlab1,infos1,cms1 = cells_to_coords(cells1)
    imlab2,infos2,cms2 = cells_to_coords(cells2)

    #find centers 2 within the cells1 and split cells1
    cms2_ = np.round(cms2).astype(int)
    cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
    imlab1_cells = [0]+[info[-1] for info in infos1]
    cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
    #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
    dic_cell2_1={}
    for cell1,cell2 in enumerate(cells2_1):
        dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
    dic_cell2_1_split = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if len(dic_cell2_1[cell])>1 and cell>0}
    cells1_split = list(dic_cell2_1_split.keys())
    imlab1_cp = imlab1.copy()
    number_of_cells_to_split = len(cells1_split)
    for cell1_split in cells1_split:
        count = np.max(imlab1_cp)+1
        cells2_to1 = dic_cell2_1_split[cell1_split]
        X1 = get_coords(imlab1,infos1,cell1_split)
        X2s = [get_coords(imlab2,infos2,cell2) for cell2 in cells2_to1]
        from scipy.spatial.distance import cdist
        X1_K = np.argmin([np.min(cdist(X1,X2),axis=-1) for X2 in X2s],0)

        for k in range(len(X2s)):
            X_ = X1[X1_K==k]
            if len(X_)>nmin:
                imlab1_cp[X_[:,0],X_[:,1]]=count+k
            else:
                #number_of_cells_to_split-=1
                pass
    imlab1_,infos1_,cms1_ = cells_to_coords(imlab1_cp)
    return imlab1_,infos1_,cms1_,number_of_cells_to_split

def converge(cells1,cells2):
    imlab1,infos1,cms1,labs1 = cells_to_coords(cells1,return_labs=True)
    imlab2,infos2,cms2 = cells_to_coords(cells2)


    #find centers 2 within the cells1 and split cells1
    cms2_ = np.round(cms2).astype(int)
    cells2_1 = imlab1[cms2_[:,1],cms2_[:,0]]
    imlab1_cells = [0]+[info[-1] for info in infos1]
    cells2_1 = [imlab1_cells.index(cl_) for cl_ in cells2_1]#reorder coords
    #[for e1,e2 in zip(np.unique(cells2_1,return_counts=True)) if e1>0]
    dic_cell2_1={}
    for cell1,cell2 in enumerate(cells2_1):
        dic_cell2_1[cell2] = dic_cell2_1.get(cell2,[])+[cell1+1]
        
    dic_cell2_1_match = {cell:dic_cell2_1[cell] for cell in dic_cell2_1 if cell>0}
    cells2_kp = [e_ for e in dic_cell2_1_match for e_ in dic_cell2_1_match[e]]
    modify_cells2 = np.setdiff1d(np.arange(len(cms2)),cells2_kp)
    imlab2_ = imlab2*0
    for cell1 in dic_cell2_1_match:
        for cell2 in dic_cell2_1_match[cell1]:
            xm,ym,sx,sy,A,icl = infos2[cell2-1]
            im_sm = imlab2[ym:ym+sy,xm:xm+sx]
            imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=labs1[cell1-1]
    count_cell = max(np.max(imlab2_),np.max(labs1))
    for cell2 in modify_cells2:
        count_cell+=1
        xm,ym,sx,sy,A,icl = infos2[cell2-1]
        im_sm = imlab2[ym:ym+sy,xm:xm+sx]
        imlab2_[ym:ym+sy,xm:xm+sx][im_sm==icl]=count_cell
    return imlab1,imlab2_
def final_segmentation(fl_dapi,
                        analysis_folder=r'X:\DCBB_human__11_18_2022_Analysis',
                        plt_val=True,
                        rescz = 4,trimz=2, resc=4,p99=None):
    segm_folder = analysis_folder+os.sep+'Segmentation'
    if not os.path.exists(segm_folder): os.makedirs(segm_folder)
    
    save_fl  = segm_folder+os.sep+os.path.basename(fl_dapi).split('.')[0]+'--'+os.path.basename(os.path.dirname(fl_dapi))+'--dapi_segm.npz'
    
    if not os.path.exists(save_fl):
        im = read_im(fl_dapi)
        #im_mid_dapi = np.array(im[-1][im.shape[1]//2],dtype=np.float32)
        im_dapi = im[-1,::rescz][trimz:-trimz]
        
        im_seg_2 = standard_segmentation(im_dapi,resc=resc,sz_min_2d=100,sz_cell=20,use_gpu=True,model='cyto2',p99=p99)
        shape = np.array(im[-1].shape)
        np.savez_compressed(save_fl,segm = im_seg_2,shape = shape)

        

    if plt_val:
        fl_png = save_fl.replace('.npz','__segim.png')
        if not os.path.exists(fl_png):
            im = read_im(fl_dapi)
            im_seg_2 = np.load(save_fl)['segm']
            shape =  np.load(save_fl)['shape']
            
            im_dapi_sm = resize(im[-1],im_seg_2.shape)
            img = np.array(im_dapi_sm[im_dapi_sm.shape[0]//2],dtype=np.float32)
            masks_ = im_seg_2[im_seg_2.shape[0]//2]
            from cellpose import utils
            outlines = utils.masks_to_outlines(masks_)
            p1,p99 = np.percentile(img,1),np.percentile(img,99.9)
            img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
            outX, outY = np.nonzero(outlines)
            imgout= np.dstack([img]*3)
            imgout[outX, outY] = np.array([1,0,0]) # pure red
            fig = plt.figure(figsize=(20,20))
            plt.imshow(imgout)
            
            fig.savefig(fl_png)
            plt.close('all')
            print("Saved file:"+fl_png)
def standard_segmentation(im_dapi,resc=2,sz_min_2d=400,sz_cell=25,use_gpu=True,model='cyto2',p99=None):
    """Using cellpose with nuclei mode"""
    from cellpose import models, io,utils
    model = models.Cellpose(gpu=use_gpu, model_type=model)
    #decided that resampling to the 4-2-2 will make it faster
    #im_dapi_3d = im_dapi[::rescz,::resc,::resc].astype(np.float32)
    chan = [0,0]
    masks_all = []
    flows_all = []
    from tqdm import tqdm
    for im in tqdm(im_dapi):
        im_ = np.array(im,dtype=np.float32)
        img = (cv2.blur(im_,(2,2))-cv2.blur(im_,(50,50)))[::resc,::resc]
        p1 = np.percentile(img,1)
        if p99 is None:
            p99 = np.percentile(img,99.9)
        img = np.array(np.clip((img-p1)/(p99-p1),0,1),dtype=np.float32)
        masks, flows, styles, diams = model.eval(img, diameter=sz_cell, channels=chan,
                                             flow_threshold=10,cellprob_threshold=-20,min_size=50,normalize=False)
        masks_all.append(utils.fill_holes_and_remove_small_masks(masks,min_size=sz_min_2d))#,hole_size=3
        flows_all.append(flows[0])
    masks_all = np.array(masks_all)

    sec_half = list(np.arange(int(len(masks_all)/2),len(masks_all)-1))
    first_half = list(np.arange(0,int(len(masks_all)/2)))[::-1]
    indexes = first_half+sec_half
    masks_all_cp = masks_all.copy()
    max_split = 1
    niter = 0
    while max_split>0 and niter<2:
        max_split = 0
        for index in tqdm(indexes):
            cells1,cells2 = masks_all_cp[index],masks_all_cp[index+1]
            imlab1_,infos1_,cms1_,no1 = resplit(cells1,cells2)
            imlab2_,infos2_,cms2_,no2 = resplit(cells2,cells1)
            masks_all_cp[index],masks_all_cp[index+1] = imlab1_,imlab2_
            max_split += max(no1,no2)
            #print(no1,no2)
        niter+=1
    masks_all_cpf = masks_all_cp.copy()
    for index in tqdm(range(len(masks_all_cpf)-1)):
        cells1,cells2 = masks_all_cpf[index],masks_all_cpf[index+1]
        cells1_,cells2_ = converge(cells1,cells2)
        masks_all_cpf[index+1]=cells2_
    return masks_all_cpf

def get_dif_or_ratio(im_sig__,im_bk__,sx=20,sy=20,pad=5,col_align=-2):
    size_ = im_sig__.shape
    imf = np.ones(size_,dtype=np.float32)
    #resc=5
    #ratios = [np.percentile(im_,99.95)for im_ in im_sig__[:,::resc,::resc,::resc]/im_bk__[:,::resc,::resc,::resc]]
    for startx in tqdm(np.arange(0,size_[2],sx)[:]):
        for starty in np.arange(0,size_[3],sy)[:]:
            startx_ = startx-pad
            startx__ = startx_ if startx_>0 else 0
            endx_ = startx+sx+pad
            endx__ = endx_ if endx_<size_[2] else size_[2]-1

            starty_ = starty-pad
            starty__ = starty_ if starty_>0 else 0
            endy_ = starty+sy+pad
            endy__ = endy_ if endy_<size_[3] else size_[3]-1

            padx_end = pad+endx_-endx__
            pady_end = pad+endy_-endy__
            padx_st = pad+startx_-startx__
            pady_st = pad+starty_-starty__

            ims___ = im_sig__[:,:,startx__:endx__,starty__:endy__]
            imb___ = im_bk__[:,:,startx__:endx__,starty__:endy__]

            txy = get_txy_small(np.max(imb___[col_align],axis=0),np.max(ims___[col_align],axis=0),sz_norm=5,delta=3,plt_val=False)
            tzy = get_txy_small(np.max(imb___[col_align],axis=1),np.max(ims___[col_align],axis=1),sz_norm=5,delta=3,plt_val=False)
            txyz = np.array([tzy[0]]+list(txy))
            #print(txyz)
            from scipy import ndimage
            for icol in range(len(imf)):
                imBT = ndimage.shift(imb___[icol],txyz,mode='nearest',order=0)
                im_rat = ims___[icol]/imBT
                #im_rat = ims___[icol]-imBT*ratios[icol]
                im_rat = im_rat[:,padx_st:-padx_end,pady_st:-pady_end]

                imf[icol,:,startx__+padx_st:endx__-padx_end,starty__+pady_st:endy__-pady_end]=im_rat
                if False:
                    plt.figure()
                    plt.imshow(np.max((im_rat),0))
                    plt.figure()
                    plt.imshow(np.max((imb___[icol,:,pad:-pad,pad:-pad]),0))
    return imf

def get_txy_small(im0_,im1_,sz_norm=10,delta=3,plt_val=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    if sz_norm>0:
        im0 -= cv2.blur(im0,(sz_norm,sz_norm))
        im1 -= cv2.blur(im1,(sz_norm,sz_norm))
    im0-=np.mean(im0)
    im1-=np.mean(im1)
    im_cor = convolve(im0[::-1,::-1],im1[delta:-delta,delta:-delta], mode='valid')
    #print(im_cor.shape)
    if plt_val:
        plt.figure()
        plt.imshow(im_cor)
    txy = np.array(np.unravel_index(np.argmax(im_cor), im_cor.shape))-delta
    return txy
def get_txyz_small(im0_,im1_,sz_norm=10,plt_val=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    im0 = norm_slice(im0,sz_norm)
    im1 = norm_slice(im1,sz_norm)
    im0-=np.mean(im0)
    im1-=np.mean(im1)
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im_cor,0))
        #print(txyz)
    txyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)-np.array(im0.shape)+1
    return txyz
    
def get_txyz_small(im0_,im1_,sz_norm=10,delta=3,plt_val=False):
    im0 = np.array(im0_,dtype=np.float32)
    im1 = np.array(im1_,dtype=np.float32)
    im0 = norm_slice(im0,sz_norm)
    im1 = norm_slice(im1,sz_norm)
    im0-=np.mean(im0)
    im1-=np.mean(im1)
    from scipy.signal import fftconvolve
    im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
    if plt_val:
        plt.figure()
        plt.imshow(np.max(im_cor,0))
        #print(txyz)
    min_ = np.array(im0.shape)-delta
    min_[min_<0]=0
    max_ = np.array(im0.shape)+delta+1
    im_cor-=np.min(im_cor)
    im_cor[tuple([slice(m,M,None)for m,M in zip(min_,max_)])]*=-1
    txyz = np.unravel_index(np.argmin(im_cor), im_cor.shape)-np.array(im0.shape)+1
    #txyz = np.unravel_index(np.argmax(im_cor_),im_cor_.shape)+delta_
    return txyz
def get_local_max(im_dif,th_fit,delta=2,delta_fit=3,dbscan=True,return_centers=False,mins=None):
    """Given a 3D image <im_dif> as numpy array, get the local maxima in cube -<delta>_to_<delta> in 3D.
    Optional a dbscan can be used to couple connected pixels with the same local maximum. 
    (This is important if saturating the camera values.)
    Returns: Xh - a list of z,x,y and brightness of the local maxima
    """
    
    z,x,y = np.where(im_dif>th_fit)
    zmax,xmax,ymax = im_dif.shape
    in_im = im_dif[z,x,y]
    keep = np.ones(len(x))>0
    for d1 in range(-delta,delta+1):
        for d2 in range(-delta,delta+1):
            for d3 in range(-delta,delta+1):
                keep &= (in_im>=im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
    z,x,y = z[keep],x[keep],y[keep]
    h = in_im[keep]
    Xh = np.array([z,x,y,h]).T
    if dbscan and len(Xh)>0:
        from scipy import ndimage
        im_keep = np.zeros(im_dif.shape,dtype=bool)
        im_keep[z,x,y]=True
        lbl, nlbl = ndimage.label(im_keep,structure=np.ones([3]*3))
        l=lbl[z,x,y]#labels after reconnection
        ul = np.arange(1,nlbl+1)
        il = np.argsort(l)
        l=l[il]
        z,x,y,h = z[il],x[il],y[il],h[il]
        inds = np.searchsorted(l,ul)
        Xh = np.array([z,x,y,h]).T
        Xh_ = []
        for i_ in range(len(inds)):
            j_=inds[i_+1] if i_<len(inds)-1 else len(Xh)
            Xh_.append(np.mean(Xh[inds[i_]:j_],0))
        Xh=np.array(Xh_)
        z,x,y,h = Xh.T
    im_centers=[]
    if delta_fit!=0 and len(Xh)>0:
        z,x,y,h = Xh.T
        z,x,y = z.astype(int),x.astype(int),y.astype(int)
        im_centers = [[],[],[],[]]
        Xft = []
        for d1 in range(-delta_fit,delta_fit+1):
            for d2 in range(-delta_fit,delta_fit+1):
                for d3 in range(-delta_fit,delta_fit+1):
                    if (d1*d1+d2*d2+d3*d3)<=(delta_fit*delta_fit):
                        im_centers[0].append((z+d1))
                        im_centers[1].append((x+d2))
                        im_centers[2].append((y+d3))
                        im_centers[3].append(im_dif[(z+d1)%zmax,(x+d2)%xmax,(y+d3)%ymax])
                        Xft.append([d1,d2,d3])
        Xft = np.array(Xft)
        sigma = 1
        norm_G = np.exp(-np.sum(Xft*Xft,axis=-1)/2./sigma/sigma)
        norm_G = norm_G/np.sum(norm_G)
        im_centers_ = np.array(im_centers)
        bk = np.min(im_centers_[-1],axis=0)
        im_centers_[-1] -= bk
        a = np.sum(im_centers_[-1],axis=0)
        hn = np.sum(im_centers_[-1]*norm_G[...,np.newaxis],axis=0)
        zc = np.sum(im_centers_[0]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        xc = np.sum(im_centers_[1]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        yc = np.sum(im_centers_[2]*im_centers_[-1],axis=0)/np.sum(im_centers_[-1],axis=0)
        Xh = np.array([zc,xc,yc,bk,a,hn,h]).T
    if return_centers:
        return Xh,np.array(im_centers)
    return Xh
from scipy.spatial.distance import cdist
def get_set(fl):
     if '_set' in fl: 
        return int(fl.split('_set')[-1].split(os.sep)[0].split('_')[0])
     else:
        return 0
from dask.array import concatenate
def concat(ims):
    shape = np.min([im.shape for im in ims],axis=0)
    ims_ = []
    for im in ims:
        shape_ = im.shape
        tupl = tuple([slice((sh_-sh)//2, -(sh_-sh)//2 if sh_>sh else None) for sh,sh_ in zip(shape,shape_)])
        ims_.append(im[tupl][np.newaxis])
    
    return concatenate(ims_)
class analysis_smFISH():
    def __init__(self,data_folder = r'X:\DCBB_human__11_18_2022',
                 save_folder = r'X:\DCBB_human__11_18_2022_Analysis',
                 H0folder=  r'X:\DCBB_human__11_18_2022\H0*',exclude_H0=True):
        self.Qfolders = glob.glob(data_folder+os.sep+'H*')
        self.H0folders = glob.glob(H0folder)
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder)
        if exclude_H0:
            self.Qfolders = [fld for fld in self.Qfolders if fld not in self.H0folders]
        self.fls_bk = np.sort([fl for H0fld in self.H0folders for fl in glob.glob(H0fld+os.sep+'*.zarr')])
        print("Found files:"+str(len(self.fls_bk)))
        print("Found hybe folders:"+str(len(self.Qfolders)))
    def set_set(self,set_=''):
        self.set_ = set_
        self.fls_bk_ = [fl for fl in self.fls_bk if set_ in fl]
    def set_fov(self,ifl,set_=None):
        if set_ is not None:
            self.set_set(set_)
        self.fl_bk = self.fls_bk_[ifl]
    def set_hybe(self,iQ):
        self.Qfolder = [qfld for qfld in self.Qfolders if self.set_ in qfld][iQ]
        self.fl = self.Qfolder+os.sep+os.path.basename(self.fl_bk)
    def get_background(self,force=False):
        ### define H0
        print('### define H0 and load background')
        if not (getattr(self,'previous_fl_bk','None')==self.fl_bk) or force:
            print("Background file: "+self.fl_bk)
            path0 =  self.fl_bk
            im0,x0,y0=read_im(path0,return_pos=True)
            self.im_bk_ = np.array(im0,dtype=np.float32)
            self.previous_fl_bk = self.fl_bk
    def get_signal(self):
        print('### load signal')
        print("Signal file: "+self.fl)
        path =  self.fl
        im,x,y=read_im(path,return_pos=True)
        self.ncols,self.szz,self.szx,self.szy = im.shape
        self.im_sig_ = np.array(im,dtype=np.float32)
    def compute_drift(self,sz=200):
        im0 = self.im_bk_[-1]
        im = self.im_sig_[-1]
        txyz,txyzs = get_txyz(im0,im,sz_norm=40,sz = sz,nelems=5,plt_val=False)
        self.txyz,self.txyzs=txyz,txyzs
        self.dic_drift = {'txyz':self.txyz,'Ds':self.txyzs,'drift_fl':self.fl_bk}
        print("Found drift:"+str(self.txyz))
    def get_aligned_ims(self):
        txyz = self.txyz
        Tref = np.round(txyz).astype(int)
        slices_bk = tuple([slice(None,None,None)]+[slice(-t_,None,None) if t_<=0 else slice(None,-t_,None) for t_ in Tref])
        slices_sig = tuple([slice(None,None,None)]+[slice(t_,None,None) if t_>=0 else slice(None,t_,None) for t_ in Tref])
        self.im_sig__ = np.array(self.im_sig_[slices_sig],dtype=np.float32)
        self.im_bk__ = np.array(self.im_bk_[slices_bk],dtype=np.float32)
    def subtract_background(self,ssub=40,s=10,plt_val=False):
        print("Reducing background...")
        self.im_ratio = get_dif_or_ratio(self.im_sig__,self.im_bk__,sx=ssub,sy=ssub,pad=5,col_align=-2)
        self.im_ration = np.array([norm_slice(im_,s=s) for im_ in self.im_ratio])
        if plt_val:
            import napari
            napari.view_image(asm.im_ration,contrast_limits=[0,0.7])
    def get_Xh(self,th = 4):
        resc=  5
        self.Xhs = [get_local_max(im_,np.std(im_[::resc,::resc,::resc])*th) for im_ in self.im_ration[:-1]]
    def check_finished_file(self):
        file_sig = self.fl
        save_folder = self.save_folder
        fov_ = os.path.basename(file_sig).split('.')[0]
        hfld_ = os.path.basename(os.path.dirname(file_sig))
        self.base_save = self.save_folder+os.sep+fov_+'--'+hfld_
        self.Xh_fl = self.base_save+'--'+'_Xh_RNAs.pkl'
        return os.path.exists(self.Xh_fl)
    def save_fits(self,icols=None,plt_val=True):
        if plt_val:
            if icols is None:
                icols =  range(self.ncols-1)
            for icol in icols:

                fig = plt.figure(figsize=(40,40))
                im_t = self.im_ration[icol]
                if False:
                    Xh = self.Xhs[icol]
                    H = Xh[:,-1]
                    vmax = np.median(np.sort(H)[-npts:])
                vmax = self.dic_th.get(icol,1)
                plt.imshow(np.max(im_t,0),vmin=0,vmax=vmax,cmap='gray')
                #plt.show()
                fig.savefig(self.base_save+'_signal-col'+str(icol)+'.png')
                plt.close('all')
        pickle.dump([self.Xhs,self.dic_drift],open(self.Xh_fl,'wb'))
def get_best_trans(Xh1,Xh2,th_h=1,th_dist = 2,return_pairs=False):
    mdelta = np.array([np.nan,np.nan,np.nan])
    if len(Xh1)==0 or len(Xh2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    X1,X2 = Xh1[:,:3],Xh2[:,:3]
    h1,h2 = Xh1[:,-1],Xh2[:,-1]
    i1 = np.where(h1>th_h)[0]
    i2 = np.where(h2>th_h)[0]
    if len(i1)==0 or len(i2)==0:
        if return_pairs:
            return mdelta,[],[]
        return mdelta
    i2_ = np.argmin(cdist(X1[i1],X2[i2]),axis=-1)
    i2 = i2[i2_]
    deltas = X1[i1]-X2[i2]
    dif_ = deltas
    bins = [np.arange(m,M+th_dist*2+1,th_dist*2) for m,M in zip(np.min(dif_,0),np.max(dif_,0))]
    hhist,bins_ = np.histogramdd(dif_,bins)
    max_i = np.unravel_index(np.argmax(hhist),hhist.shape)
    #plt.figure()
    #plt.imshow(np.max(hhist,0))
    center_ = [(bin_[iM_]+bin_[iM_+1])/2. for iM_,bin_ in zip(max_i,bins_)]
    keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
    center_ = np.mean(dif_[keep],0)
    for i in range(5):
        keep = np.all(np.abs(dif_-center_)<=th_dist,-1)
        center_ = np.mean(dif_[keep],0)
    mdelta = center_
    keep = np.all(np.abs(deltas-mdelta)<=th_dist,1)
    if return_pairs:
        return mdelta,Xh1[i1[keep]],Xh2[i2[keep]]
    return mdelta
    
def norm_im_med(im,im_med):
    if len(im_med)==2:
        return (im.astype(np.float32)-im_med[0])/im_med[1]
    else:
        return im.astype(np.float32)/im_med
def read_im(path,return_pos=False,th=300):
    import zarr,os
    from dask import array as da
    dirname = os.path.dirname(path)
    fov = os.path.basename(path).split('_')[-1].split('.')[0]
    #print("Bogdan path:",path)
    file_ = dirname+os.sep+fov+os.sep+'data'
    #image = zarr.load(file_)[1:]
    image = da.from_zarr(file_)[1:]
    #image.dtype
    if str(image.dtype)=='uint8':
        image = image.astype(np.uint16)**2
    if th is not None: ### This fixes a bug with lazy camera
        for i_,fr in enumerate(image):
            if np.median(np.array(fr))>th:
                break
        image = image[i_:]    
    shape = image.shape
    #nchannels = 4
    xml_file = os.path.dirname(path)+os.sep+os.path.basename(path).split('.')[0]+'.xml'
    if os.path.exists(xml_file):
        txt = open(xml_file,'r').read()
        tag = '<z_offsets type="string">'
        zstack = txt.split(tag)[-1].split('</')[0]
        
        tag = '<stage_position type="custom">'
        x,y = eval(txt.split(tag)[-1].split('</')[0])
        
        nchannels = int(zstack.split(':')[-1])
        nzs = (shape[0]//nchannels)*nchannels
        image = image[:nzs].reshape([shape[0]//nchannels,nchannels,shape[-2],shape[-1]])
        image = image.swapaxes(0,1)
    shape = image.shape
    if return_pos:
        return image,x,y
    return image


def linear_flat_correction(ims,fl=None,reshape=True,resample=4,vec=[0.1,0.15,0.25,0.5,0.75,0.9]):
    #correct image as (im-bM[1])/bM[0]
    #ims=np.array(ims)
    if reshape:
        ims_pix = np.reshape(ims,[ims.shape[0]*ims.shape[1],ims.shape[2],ims.shape[3]])
    else:
        ims_pix = np.array(ims[::resample])
    ims_pix_sort = np.sort(ims_pix[::resample],axis=0)
    ims_perc = np.array([ims_pix_sort[int(frac*len(ims_pix_sort))] for frac in vec])
    i1,i2=np.array(np.array(ims_perc.shape)[1:]/2,dtype=int)
    x = ims_perc[:,i1,i2]
    X = np.array([x,np.ones(len(x))]).T
    y=ims_perc
    a = np.linalg.inv(np.dot(X.T,X))
    cM = np.swapaxes(np.dot(X.T,np.swapaxes(y,0,-2)),-2,1)
    bM = np.swapaxes(np.dot(a,np.swapaxes(cM,0,-2)),-2,1)
    if fl is not None:
        folder = os.path.dirname(fl)
        if not os.path.exists(folder):
            os.makedirs(folder)
        pickle.dump(bM,open(fl,'wb'))
    return bM 
def compose_mosaic(ims,xs_um,ys_um,ims_c=None,um_per_pix=0.108333,rot = 0,return_coords=False):
    dtype = np.float32
    im_ = ims[0]
    szs = im_.shape
    sx,sy = szs[-2],szs[-1]
    ### Apply rotation:
    theta=-np.deg2rad(rot)
    xs_um_ = np.array(xs_um)*np.cos(theta)-np.array(ys_um)*np.sin(theta)
    ys_um_ = np.array(ys_um)*np.cos(theta)+np.array(xs_um)*np.sin(theta)
    ### Calculate per pixel
    xs_pix = np.array(xs_um_)/um_per_pix
    xs_pix = np.array(xs_pix-np.min(xs_pix),dtype=int)
    ys_pix = np.array(ys_um_)/um_per_pix
    ys_pix = np.array(ys_pix-np.min(ys_pix),dtype=int)
    sx_big = np.max(xs_pix)+sx+1
    sy_big = np.max(ys_pix)+sy+1
    dim = [sx_big,sy_big]
    if len(szs)==3:
        dim = [szs[0],sx_big,sy_big]

    if ims_c is None:
        if len(ims)>25:
            try:
                ims_c = linear_flat_correction(ims,fl=None,reshape=False,resample=1,vec=[0.1,0.15,0.25,0.5,0.65,0.75,0.9])
            except:
                imc_c = np.median(ims,axis=0)
        else:
            ims_c = np.median(ims,axis=0)

    im_big = np.zeros(dim,dtype = dtype)
    sh_ = np.nan
    for i,(im_,x_,y_) in enumerate(zip(ims,xs_pix,ys_pix)):
        if ims_c is not None:
            if len(ims_c)==2:
                im_coef,im_inters = np.array(ims_c,dtype = 'float32')
                im__=(np.array(im_,dtype = 'float32')-im_inters)/im_coef
            else:
                ims_c_ = np.array(ims_c,dtype = 'float32')
                im__=np.array(im_,dtype = 'float32')/ims_c_*np.median(ims_c_)
        else:
            im__=np.array(im_,dtype = 'float32')
        im__ = np.array(im__,dtype = dtype)
        im_big[...,x_:x_+sx,y_:y_+sy]=im__
        sh_ = im__.shape
    if return_coords:
        return im_big,xs_pix+sh_[-2]/2,ys_pix+sh_[-1]/2
    return im_big
import cv2

def get_tiles(im_3d,size=256,delete_edges=False):
    sz,sx,sy = im_3d.shape
    if not delete_edges:
        Mz = int(np.ceil(sz/float(size)))
        Mx = int(np.ceil(sx/float(size)))
        My = int(np.ceil(sy/float(size)))
    else:
        Mz = np.max([1,int(sz/float(size))])
        Mx = np.max([1,int(sx/float(size))])
        My = np.max([1,int(sy/float(size))])
    ims_dic = {}
    for iz in range(Mz):
        for ix in range(Mx):
            for iy in range(My):
                ims_dic[(iz,ix,iy)]=ims_dic.get((iz,ix,iy),[])+[im_3d[iz*size:(iz+1)*size,ix*size:(ix+1)*size,iy*size:(iy+1)*size]] 
    return ims_dic
def norm_slice(im,s=50):
    im_=im.astype(np.float32)
    return np.array([im__-cv2.blur(im__,(s,s)) for im__ in im_],dtype=np.float32)

def get_txyz(im_dapi0,im_dapi1,sz_norm=40,sz = 200,nelems=5,plt_val=False):
    """
    Given two 3D images im_dapi0,im_dapi1, this normalizes them by subtracting local background (gaussian size sz_norm)
    and then computes correlations on <nelemes> blocks with highest  std of signal of size sz
    It will return median value and a list of single values.
    """
    im_dapi0 = np.array(im_dapi0,dtype=np.float32)
    im_dapi1 = np.array(im_dapi1,dtype=np.float32)
    im_dapi0_ = norm_slice(im_dapi0,sz_norm)
    im_dapi1_ = norm_slice(im_dapi1,sz_norm)
    dic_ims0 = get_tiles(im_dapi0_,size=sz,delete_edges=True)
    dic_ims1 = get_tiles(im_dapi1_,size=sz,delete_edges=True)
    keys = list(dic_ims0.keys())
    best = np.argsort([np.std(dic_ims0[key]) for key in keys])[::-1]
    txyzs = []
    im_cors = []
    for ib in range(min(nelems,len(best))):
        im0 = dic_ims0[keys[best[ib]]][0].copy()
        im1 = dic_ims1[keys[best[ib]]][0].copy()
        im0-=np.mean(im0)
        im1-=np.mean(im1)
        from scipy.signal import fftconvolve
        im_cor = fftconvolve(im0[::-1,::-1,::-1],im1, mode='full')
        if plt_val:
            plt.figure()
            plt.imshow(np.max(im_cor,0))
            #print(txyz)
        txyz = np.unravel_index(np.argmax(im_cor), im_cor.shape)-np.array(im0.shape)+1
        
        im_cors.append(im_cor)
        txyzs.append(txyz)
    txyz = np.median(txyzs,0).astype(int)
    return txyz,txyzs

from tqdm import tqdm
from tqdm import tqdm
def get_mosaic_image_T(data_fld,resc = 4,icol = 1,frame = 20,force=False,rescz=2):
    save_fld = os.path.dirname(data_fld)+os.sep+'mosaics'
    if not os.path.exists(save_fld): os.makedirs(save_fld)
    fl_save = save_fld+os.sep+os.path.basename(data_fld)+'_frame'+str(frame)+'_col'+str(icol)+'.tiff'
    print(fl_save)
    if not os.path.exists(fl_save) or force:
        fls_ = np.sort(glob.glob(data_fld+r'\*.zarr'))#[:100]
        ims,xs_um,ys_um=[],[],[]

        for fl in tqdm(fls_[:]):
            im,x,y = read_im(fl,return_pos=True)
            
            if str(frame).lower()=='all':
                ims.append(np.array(np.max(im[icol][::rescz,::resc,::resc],axis=0),dtype=np.float32))
            else:
                ims.append(np.array(im[icol][frame,::resc,::resc],dtype=np.float32))
            xs_um.append(x)
            ys_um.append(y)
        ims_ = [im[::-1,::-1] for im in ims]
        im_big,xs,ys = compose_mosaic(ims_,xs_um,ys_um,ims_c=None,
                                      um_per_pix=0.1083333*resc,
                                      rot = 0,return_coords= True)
        
        import tifffile
        tifffile.imwrite(fl_save,im_big)
        
        
        resc_ = 3
        data_fld = os.path.dirname(fls_[0])
        fig = plt.figure(figsize=(30,30))
        im__ = im_big[::resc_,::resc_]
        #if vmax is None:
        vmax = np.percentile(im__[im__>0],99.)
        vmin=np.percentile(im__[im__>0],1.)
        plt.imshow(im_big.T[::resc_,::resc_],vmin=vmin,vmax=vmax,cmap='gray')
        #fig.savefig(fl_save)
        for x_,y_,fl_ in zip(xs,ys,fls_):
            ifov = fl_.split('_')[-1].split('.')[0]
            plt.text(x_/resc_,y_/resc_,ifov,color='r')
        plt.xticks([])
        plt.yticks([])
        fl_save = fl_save.replace('.tiff','_annot.png')
        fl_saveT = fl_save.replace('.tiff','.npz')
        np.savez(fl_saveT,fls=fls_,xs=xs,ys=ys,xs_um=xs_um,ys_um=ys_um)
        print(fl_save)
        fig.savefig(fl_save)
        plt.close('all')
