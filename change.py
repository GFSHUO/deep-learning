from keras.preprocessing.image import ImageDataGenerator

path = 'C:/Users/36327/Desktop/人脸/yhy' # 类别子文件夹的上一级

dst_path = 'C:/Users/36327/Desktop/人脸/yhy/face6'

# 　图片生成器

datagen = ImageDataGenerator(

    rotation_range=25,

    width_shift_range=0.03,

    height_shift_range=0.03,

    shear_range=0.02,

    horizontal_flip=True,

    fill_mode='nearest',

    zoom_range=0.2,

)

gen = datagen.flow_from_directory(

                        path,

                        target_size=(128, 128),

                        batch_size=200,

                        save_to_dir=dst_path,#生成后的图像保存路径

                        save_prefix='xx',

                        save_format='jpg')



for i in range(6):

    gen.next()
