import cv2 as cv
import wx
import tqdm
import os
import numpy as np
from glob import glob
from skew import rotate
from multiprocessing import Pool


def skew(img_path):
    img_name = os.path.basename(img_path)
    img_size = os.path.getsize(img_path)

    if 100 < img_size:
        img_read = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img_read is None:
            print("Invalid image:", img_name)
        else:            
            scale_percent = 10 
            width = int(img_read.shape[1] * scale_percent / 100)
            height = int(img_read.shape[0] * scale_percent / 100)
            dim = (width, height)
            img_temp = cv.resize(img_read, dim, interpolation=cv.INTER_AREA)
            img_skewed, h_ri, h_rf, v_ri, v_rf = rotate(img_temp, img_read)
            
            img_skewed = img_skewed[h_ri:h_rf, v_ri:v_rf]
            
            # DEBUG
            cv.namedWindow("img_straighted", cv.WINDOW_NORMAL)
            cv.namedWindow("img_original", cv.WINDOW_NORMAL)
            cv.moveWindow("img_straighted", 1100, 0)
            cv.moveWindow("img_original", 0, 0)
            cv.imshow("img_straighted", img_skewed)
            cv.imshow("img_original", img_read)
            cv.waitKey()
            cv.destroyAllWindows()


def get_path():
    app = wx.App()
    img_dir = ""
    dialog = wx.DirDialog(None, "Selecciona la carpeta de imágenes JPG a procesar:",
                          style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON)
    if dialog.ShowModal() == wx.ID_OK:
        img_dir = (dialog.GetPath() + "/")
    else:
        print("    │No se seleccionó ninguna carpeta.")
        input("    │Proceso terminado. Presiona ENTER para salir.\n")
        exit()
    dialog.Destroy()
    return img_dir


def main(name):
    dir_path = get_path()
    raw_img_list = glob(dir_path + "/*.jpg")
    for r in raw_img_list:
        print(r)
        skew(r)


if __name__ == '__main__':
    main('PyCharm')
