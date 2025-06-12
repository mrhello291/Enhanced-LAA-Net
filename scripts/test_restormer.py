import os
import sys
from restormer.model import RestormerDerainer 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StormSight Model Pipeline")
    parser.add_argument("--input", type=str, default="./data/input", help="Directory with input images")
    parser.add_argument("--output", type=str, default="./data/output", help="Directory to save output results")
    args = parser.parse_args()

    print("Running the model pipeline...")

    os.makedirs("./data/denoised", exist_ok=True)

    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]


    derainer = RestormerDerainer()

    for img_file in image_files:
        input_image_path = os.path.join(args.input, img_file)
        output_image_path = os.path.join("./data/denoised", f"denoised_{img_file}")


        derainer.derain_image(input_image_path, output_image_path)
        #derain_image(input_image_path, output_image_path)
