import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from skimage.transform import resize
from skimage.io import imsave, imshow_collection, imshow
from skimage.draw import line, rectangle, rectangle_perimeter

def np_to_torch(img_np):
    '''
    np                torch
    ---                ---
    512 x 512 x 3 ->  1 x 3 x 512 x 512
    512 x 512     ->  1 x 1 x 512 x 512
    '''
    assert len(img_np.shape) in [2,3]
    if len(img_np.shape) == 3:
        return torch.from_numpy(img_np).permute(2,0,1)[None, :]
    elif len(img_np.shape) == 2:
        return torch.from_numpy(img_np)[None, None, :]
    else:
        assert False
        
def torch_to_np(img_var):
    '''
    torch                 np
    ---                   ---
    1 x 3 x 512 x 512 ->  512 x 512 x 3
    1 x 1 x 512 x 512 ->  512 x 512
    '''
    assert len(img_var.shape) == 4
    assert img_var.shape[1] in [1,3]
    if img_var.shape[1] == 3:
        return img_var.detach()[0].cpu().permute(1,2,0).numpy()
    elif img_var.shape[1] == 1:
        return img_var.detach()[0,0].cpu().numpy()
    else:
        assert False

def im2tensor(x, grad=False):
    t = torch.from_numpy(x).float()
    t.requires_grad = grad
    return t


def plot_result(gt, noisy, result, FOCUS=None, show=False, save_name=None):
    fig, ax = plt.subplots(2, 1)
    ims = np.hstack([gt, noisy, result])
    focussed_ims = np.hstack(
        [FOCUS(gt), FOCUS(noisy), FOCUS(result)])
    ax[0].imshow(np.clip(ims, 0, 1), cmap='gray')
    ax[1].imshow(np.clip(focussed_ims, 0, 1), cmap='gray')
    if show:
        plt.show()
    if save_name:
        plt.savefig(save_name)
    plt.close()

def plot_grid(
        imgs,
        FOCUS=None,
        ZOOM=None,
        number_of_rows=1,
        margin=None,
        show=False,
        save_name=None,
        plot1d=None,
        dpi=300):
    """Grid plotter function for

    Args:
        imgs ([type]): Array of Numpy ndarrays dim should be 2D
        FOCUS (func, optional): Focus function creates
            a rectangle and show zooomed version there. Defaults to None.
        ZOOM (func, optional): Zoom function creates a rectange and 
            crop the images to that area. Defaults to None.
        number_of_rows (int, optional): Number of row for images, 
            columns will be calculated automatically. Defaults to 1.
        margin (int, optional): number of pixel for the margin. Defaults to None.
        show (bool, optional): show the image grid or not. Defaults to False.
        save_name (str, optional): grid save img directory 
            if not defined images will not be saved . Defaults to None.
        plot1d (int, optional): 1d plot position, Defauls None.
        dpi (int, optional): DPI of save image. Defaults to 300.
    """
    updated_imgs = []

    ### FOCUS and Zoom
    if FOCUS:
        for img in imgs:
            # find focus
            focused_img, coor = FOCUS(img)
            f_size_x, f_size_y = focused_img.shape
            focused_img = resize(focused_img, (f_size_x*2, f_size_y*2))
            f_size_x, f_size_y = focused_img.shape
            size_x, size_y = img.shape
            uimg = img.copy()
            # draw focus rect
            rr, cc = rectangle_perimeter(
                coor[0:2], 
                coor[2:4]
            )
            uimg[rr, cc] = 1
            # draw big focus rect
            rr, cc = rectangle_perimeter(
                (size_x-f_size_x, size_y-f_size_y),
                (size_x-2, size_y-2),
            )
            uimg[size_x-f_size_x:size_x, size_y-f_size_y:size_y] = focused_img
            uimg[rr, cc] = 1
            # append img
            updated_imgs.append(uimg)
    elif ZOOM:
        for img in imgs:
            focused_img, coor = ZOOM(img)
            # find focus
            updated_imgs.append(focused_img)
    else:
        updated_imgs = imgs

    # GRID SHAPE
    number_of_columns = len(imgs) // number_of_rows
    updated_imgs = np.vstack(
        [np.hstack(updated_imgs[i:i+number_of_columns]) for i in range(0, len(imgs), number_of_columns)  ]
    )

    # Save and Show
    ims = np.clip(updated_imgs, 0, 1) * 255
    ims = ims.astype(np.uint8)
    
    # 1D plot
    if plot1d:
        h, w = ims.shape
        #plt.figure(figsize=(30, 30))
        fig, (ax1,ax2) =  plt.subplots(nrows=2, sharex=True)
        plt.subplots_adjust(hspace = .001)
        #ax2.set_aspect(4.0/len(imgs))
        ax2.plot(ims[plot1d,:])
        th_h = h//20
        th_w = (w//number_of_columns) // 20
        for th in range(-th_h//2, th_h//2):
            ims[plot1d+th,[i for i in range(0, w, th_w)]] = 255
            ims[plot1d+th,[i+1 for i in range(0, w, th_w)]] = 255
        ax1.imshow(ims, cmap='gray')
        plt.savefig("plt_" + save_name, 
                    dpi=dpi, bbox_inches="tight")

    if show:
        plt.figure()
        imshow(ims)
        plt.show()
    if save_name:
        imsave(save_name, ims)
    # if focus_save_name:
    #     imsave(focus_save_name, [fo])
