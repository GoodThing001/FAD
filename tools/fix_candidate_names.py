import os

root = "/home/hzeng/project/FAD_CLEAN/data/candidates"
for f in os.listdir(root):
    if "\\" in f:
        os.rename(os.path.join(root, f), os.path.join(root, f.split("\\")[-1]))
print(sorted(os.listdir(root))[:5])
print("n csv:", len([f for f in os.listdir(root) if f.endswith(".csv")]))
