from natsort import natsorted
from torch.utils.data import Dataset
import albumentations as A

class AugDataset(Dataset):
    def __init__(self, root, opt, distortion=None):
        self.opt = opt
        self.distortion = distortion
        self.augment = distortion is not None  # Set to True if a distortion is provided

        # Choose the transformation based on the distortion parameter
        transformations = {
            'contrast': A.RandomContrast(limit=0.2, p=1),
            'flip': A.HorizontalFlip(p=1),
            'brightness': A.RandomBrightnessContrast(brightness_limit=0.2, p=1),
            'gaussian_blur': A.GaussianBlur(blur_limit=(3, 7), p=1),
            'motion_blur': A.MotionBlur(blur_limit=(3, 7), p=1),
            'median_blur': A.MedianBlur(blur_limit=3, p=1),
            'compression': A.ImageCompression(quality_lower=80, quality_upper=100, p=1), 
            'gird_distort': A.GridDistortion(num_steps=5, distort_limit=0.3, p=1), 
            'optical_distort': A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1), 
        }
        self.transform = A.Compose([transformations[distortion]]) if distortion in transformations else A.NoOp()

        # Collect image paths
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                if ext.lower() in ['.jpg', '.jpeg', '.png']:
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        try:
            img = Image.open(img_path).convert('RGB' if self.opt.rgb else 'L')
            if self.augment:
                img = np.array(img)
                img = self.transform(image=img)['image']
                img = Image.fromarray(img)
        except IOError:
            print(f'Corrupted image for {index}')
            img = Image.new('RGB' if self.opt.rgb else 'L', (self.opt.imgW, self.opt.imgH))

        return img, img_path
