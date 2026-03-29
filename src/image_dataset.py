import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PhishingImageDataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.samples = []
        self.train = train

        # -------- AUGMENTATION --------
        if self.train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        # ------------------------------

        for label, folder in enumerate(["legit", "phishing"]):
            folder_path = os.path.join(root_dir, folder)

            for img_name in os.listdir(folder_path):

                # 🔥 Skip non-image files (important)
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue

                self.samples.append(
                    (os.path.join(folder_path, img_name), label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        # 🔥 CRITICAL FIX: return filename
        filename = os.path.basename(path)

        return image, label, filename