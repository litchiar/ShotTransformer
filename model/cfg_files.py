from tools.load_dataset import get_project_path
import os
cfg_files={
    'MViTv2': os.path.join(get_project_path(),"yaml\MVITv2_S_16x4.yaml"),

}
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

if __name__ == '__main__':
    print(cfg_files)