import cv2
import os

imageroot = "/media/HardDisk_B/ym/CoSOD/MSRC/img/"
heatmaproot = "/media/HardDisk_B/ym/Test/prediction52/MSRC"
saveroot = "/media/HardDisk_B/ym/Test/heatMap/MSRC"

for subdir in os.listdir(imageroot):
    subPath = os.path.join(imageroot, subdir)
    save_path = os.path.join(saveroot, subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for fileName in os.listdir(subPath):
        imagePath = os.path.join(subPath, fileName)   
        img = cv2.imread(imagePath)
        print(img)

        heatmapPath = os.path.join(heatmaproot, subdir, fileName[:-4]+"_hm.png")
        heatmap = cv2.imread(heatmapPath)
        print(heatmapPath)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap*0.5 + img*0.5
        cv2.imwrite(os.path.join(save_path, fileName), superimposed_img)