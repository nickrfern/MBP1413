print('python start')

from sklearn.model_selection import train_test_split
import pandas as pd
import time
import os

print('packge loaded')

# ####local test
# cellname_path = './test_data/pbmc68k_cellnames.csv'
# pbmc68k_raw_x_path = './test_data/pbmc68k_raw_x.csv'
# pbmc68k_y_path = './test_data/pbmc68k_y.csv'

####HPF
cellname_path = '/hpf/largeprojects/tabori/users/yuan/mbp1413/data/raw_data/pbmc68k_cellnames.csv'
pbmc68k_raw_x_path = '/hpf/largeprojects/tabori/users/yuan/mbp1413/data/raw_data/pbmc68k_raw_x.csv'
pbmc68k_y_path = '/hpf/largeprojects/tabori/users/yuan/mbp1413/data/raw_data/pbmc68k_y.csv'

print('script start')
load_start = time.time()
cell_name = pd.read_csv(cellname_path)
pbmc68k_y = pd.read_csv(pbmc68k_y_path)
pbmc68k_raw_x = pd.read_csv(pbmc68k_raw_x_path)
load_end = time.time()
print(f'python load all data time cost {load_end-load_start} s')

print('train:val:test = 7:1:2')

name_temp,name_test,x_temp,x_test,y_temp,y_test=train_test_split(
    cell_name,pbmc68k_raw_x.set_index('Unnamed: 0').T,pbmc68k_y,
    random_state=42,
    test_size=0.2,
    stratify=pbmc68k_y['x']
    )

# ####local test
# x_test.T.to_csv('./test_data/test/pbmc68k_raw_x_test.csv')
# del x_test
# name_test.set_index('Unnamed: 0').to_csv('./test_data/test/cell_name_test_cv.csv')
# del name_test
# y_test.set_index('Unnamed: 0').to_csv('./test_data/test/pbmc68k_y_test_cv.csv')
# del y_test
# print('spliting')

####HPF
x_test.T.to_csv('/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/raw_data/pbmc68k_raw_x_test_cv.csv')
del x_test
name_test.set_index('Unnamed: 0').to_csv('/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/raw_data/cell_name_test_cv.csv')
del name_test
y_test.set_index('Unnamed: 0').to_csv('/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/test_cv/raw_data/pbmc68k_y_test_cv.csv')
del y_test
print('spliting')


for fold_n in range(5):
    name_train,name_val,x_train,x_val,y_train,y_val=train_test_split(
        name_temp,x_temp,y_temp,
        random_state=fold_n,
        test_size=(1/8),
        stratify=y_temp['x']
        )

    print(f'exporting{fold_n+1}')
    
    # ####local test
    # ###test
    # train_val_path = f'/Users/yuanchang/Documents/MBP/AI_course/code/sc/test_data/train_val_cv{fold_n+1}'
    # os.makedirs(f'{train_val_path}/train/')
    # os.makedirs(f'{train_val_path}/validation/')
    # name_train.set_index('Unnamed: 0').to_csv(f'{train_val_path}/train/cell_name_train_{fold_n+1}.csv')
    # name_val.set_index('Unnamed: 0').to_csv(f'{train_val_path}/validation/cell_name_validation_{fold_n+1}.csv')

    # x_train.T.to_csv(f'{train_val_path}/train/pbmc68k_raw_x_train_{fold_n+1}.csv')
    # x_val.T.to_csv(f'{train_val_path}/validation/pbmc68k_raw_x_validation_{fold_n+1}.csv')

    # y_train.set_index('Unnamed: 0').to_csv(f'{train_val_path}/train/pbmc68k_y_train_{fold_n+1}.csv')
    # y_val.set_index('Unnamed: 0').to_csv(f'{train_val_path}/validation/pbmc68k_yvalidation_{fold_n+1}.csv')


    ####HPF
    train_val_path = f'/hpf/largeprojects/tabori/users/yuan/mbp1413/data/train_test_val/train_val_cv{fold_n+1}'
    os.makedirs(f'{train_val_path}/train/raw_data')
    os.makedirs(f'{train_val_path}/validation/raw_data')
    name_train.set_index('Unnamed: 0').to_csv(f'{train_val_path}/train/raw_data/cell_name_train.csv')
    name_val.set_index('Unnamed: 0').to_csv(f'{train_val_path}/validation/raw_data/cell_name_validation.csv')

    x_train.T.to_csv(f'{train_val_path}/train/raw_data/pbmc68k_raw_x_train.csv')
    x_val.T.to_csv(f'{train_val_path}/validation/raw_data/pbmc68k_raw_x_validation.csv')

    y_train.set_index('Unnamed: 0').to_csv(f'{train_val_path}/train/raw_data/pbmc68k_y_train.csv')
    y_val.set_index('Unnamed: 0').to_csv(f'{train_val_path}/validation/raw_data/pbmc68k_yvalidation.csv')

