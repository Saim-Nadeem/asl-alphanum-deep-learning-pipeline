import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# -------------- Preprocessing Utilities --------------
def convert_to_rgb_array(image):
    if image.ndim == 2:
        return np.stack([image]*3, axis=-1).astype(np.uint8)
    elif image.shape[2] == 4:
        return image[:, :, :3]
    return image

def gaussian_blur(img, kernel_size=5, sigma=1.0):
    def gaussian(x, y):
        return (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2)/(2*sigma**2))

    k = kernel_size // 2
    kernel = np.array([[gaussian(x, y) for x in range(-k, k+1)] for y in range(-k, k+1)])
    kernel /= np.sum(kernel)

    padded = np.pad(img, ((k, k), (k, k), (0, 0)), mode='reflect')
    blurred = np.zeros_like(img)

    for c in range(3):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+kernel_size, j:j+kernel_size, c]
                blurred[i, j, c] = np.sum(region * kernel)
    return blurred.astype(np.uint8)

def sharpen_image(img):
    kernel = np.array([[0, -1,  0],
                       [-1,  5, -1],
                       [0, -1,  0]])
    k = 1
    padded = np.pad(img, ((k, k), (k, k), (0, 0)), mode='reflect')
    sharp = np.zeros_like(img)
    for c in range(3):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+3, j:j+3, c]
                sharp[i, j, c] = np.clip(np.sum(region * kernel), 0, 255)
    return sharp.astype(np.uint8)

def rgb_to_ycrcb(image):
    img = image.astype(np.float32)
    Y = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    Cr = (img[:,:,0] - Y) * 0.713 + 128
    Cb = (img[:,:,2] - Y) * 0.564 + 128
    return np.stack([Y, Cr, Cb], axis=-1).astype(np.uint8)

def strict_skin_mask(img):
    ycrcb = rgb_to_ycrcb(img)
    Cr = ycrcb[:,:,1]
    Cb = ycrcb[:,:,2]
    mask = np.logical_and.reduce((Cr > 135, Cr < 175, Cb > 90, Cb < 135))
    return mask.astype(np.uint8) * 255

def remove_small_components(mask, min_size_ratio=0.02):
    h, w = mask.shape
    total_pixels = h * w
    visited = np.zeros_like(mask, dtype=bool)
    output = np.zeros_like(mask)

    def dfs(i, j, label_map):
        stack = [(i, j)]
        count = 0
        while stack:
            x, y = stack.pop()
            if not (0 <= x < h and 0 <= y < w): continue
            if visited[x, y] or mask[x, y] == 0: continue
            visited[x, y] = True
            label_map.append((x, y))
            count += 1
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((x+dx, y+dy))
        return count

    for i in range(h):
        for j in range(w):
            if not visited[i, j] and mask[i, j] == 255:
                label_map = []
                size = dfs(i, j, label_map)
                if size > total_pixels * min_size_ratio:
                    for x, y in label_map:
                        output[x, y] = 255
    return output

def apply_mask_on_black_background(original, mask):
    output = np.zeros_like(original)
    for c in range(3):
        output[:, :, c] = np.where(mask == 255, original[:, :, c], 0)
    return output

def normalize_image(img):
    return img.astype(np.float32) / 255.0

def preprocess_hand_image_pil(pil_image):
    img = convert_to_rgb_array(np.array(pil_image))
    blurred = gaussian_blur(img, kernel_size=5, sigma=1.2)
    sharpened = sharpen_image(blurred)
    skin = strict_skin_mask(sharpened)
    cleaned = remove_small_components(skin)
    final_masked = apply_mask_on_black_background(sharpened, cleaned)
    return Image.fromarray(final_masked)

# -------------- Batch Processing Code --------------

input_root = 'asl_alphabet_train'
output_root = 'Preprocessed data'

if not os.path.exists(output_root):
    os.makedirs(output_root)

def process_image(input_file_path, output_file_path):
    try:
        with Image.open(input_file_path) as img:
            processed_img = preprocess_hand_image_pil(img)
            processed_img.save(output_file_path, 'JPEG')
    except Exception as e:
        print(f"Error processing file: {input_file_path} - {e}")

def process_folder(input_subfolder_path, output_subfolder_path):
    if not os.path.exists(output_subfolder_path):
        os.makedirs(output_subfolder_path)

    all_filenames = [filename for filename in os.listdir(input_subfolder_path)
                     if os.path.isfile(os.path.join(input_subfolder_path, filename))]

    selected_filenames = all_filenames[:1000]

    files = [(os.path.join(input_subfolder_path, filename),
              os.path.join(output_subfolder_path, os.path.splitext(filename)[0] + '.jpg'))
             for filename in selected_filenames]

    batch_size = 16
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(lambda args: process_image(*args), batch)

def main():
    for subfolder in os.listdir(input_root):
        input_subfolder_path = os.path.join(input_root, subfolder)
        if not os.path.isdir(input_subfolder_path):
            continue
        output_subfolder_path = os.path.join(output_root, subfolder)
        process_folder(input_subfolder_path, output_subfolder_path)

if __name__ == '__main__':
    main()