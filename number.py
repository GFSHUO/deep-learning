import os
from PIL import Image, ImageEnhance




path = "C:/Users/36327/Desktop/face55"
save_path = "C:/Users/36327/Desktop/face55"
file_list = os.listdir(path)
for file in file_list:
    all_jpg = os.path.join(path , file)
    jpg = Image.open(all_jpg)
    '''brightness = ImageEnhance.Brightness(jpg)
    im_brightness = brightness.enhance(1.6)
    im_brightness.save(os.path.join(save_path , file))'''

    '''enh_sharpness = ImageEnhance.Sharpness(jpg)
    sharpness = 1.5
    enh_sharpness = enh_sharpness.enhance(sharpness)
    enh_sharpness.save(os.path.join(save_path , file))'''
    '''enh_col = ImageEnhance.Color(jpg )
    color = 2
    image_colored = enh_col.enhance(color)
    image_colored.save(os.path.join(save_path , file))'''
    enh_con = ImageEnhance.Contrast(jpg)
    contrast = 2
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.save(os.path.join(save_path , file))