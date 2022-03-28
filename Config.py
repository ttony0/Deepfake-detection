# BATCH_SIZE = 4
DATA_GLOB_PATH = {'REAL': ['..\\data\\Celeb-real\\*.mp4'],
                  'FAKE': ['..\\data\\Celeb-synthesis\\*.mp4']}
RESULT_GLOB_PATH = {'REAL': ['..\\data\\Processed_Data\\REAL\\*.mp4'],
                    'FAKE': ['..\\data\\Processed_Data\\FAKE\\*.mp4']}
RESULT_PATH = {'REAL': '..\\data\\Processed_Data\\REAL',
               'FAKE': '..\\data\\Processed_Data\\FAKE'}
MODEL_SAVE_DIR = '.\\model\\'


REAL = 1
FAKE = 0
timeF = 6
BATCH_SIZE = 1
sequence_length = 15

p = 0.8
lr = 1e-5
epochs = 5

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]