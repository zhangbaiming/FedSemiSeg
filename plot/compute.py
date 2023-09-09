from glob import glob

path1 = '/home/zbm/desktop/data/2023-01-10 18-30-36.145777/*_mask_*.png'
path2 = '/home/zbm/desktop/data/2023-01-11 03-01-27.304278/*_mask_*.png'

list1 = sorted(glob(path1))
list2 = sorted(glob(path2))

for a, b in zip(list1, list2):
    d1 = float(a[-10:-4])
    d2 = float(b[-10:-4])
    if d2 > 0.8 and d2-d1 > 0.3:
        print(a.split('/')[-1].split('_')[0])
