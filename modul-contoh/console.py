import cv2
import glob
import argparse
from pathlib import Path
from function.fruits_counter import counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    input = str(args.input)
    output = str(args.output)

    filenames = [img for img in glob.glob(input+"/*.*")]

    for imgfile in filenames:
        img_path = f'{imgfile}'
        img_split = Path(img_path).stem
        print(img_split)

        img = cv2.imread(imgfile)
        new_img, num_obj = counter(img)
        print("jumlah buah: "+str(num_obj))
        cv2.imwrite(output+"/"+img_split+".png", new_img)
