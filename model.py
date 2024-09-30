from PIL import Image
from torchvision import transforms
import torch
from transformers import AutoModelForImageClassification
import numpy as np
from safetensors.torch import safe_open


class MyModel():
    def __init__(self):
        self.model_loaded = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.loaded_model = None
        self.request = None

    def setRequest(self, request):
        self.request = request

    async def load_model(self, root_url):
        if self.model_loaded:
            return self.loaded_model
        model = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", ignore_mismatched_sizes=True, num_labels=10)
        model_weights = {}
        with safe_open("models/model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                model_weights[key] = f.get_tensor(key)
        model.load_state_dict(model_weights)
        self.loaded_model = model
        self.model_loaded = True
        return self.loaded_model

    def get_loaded_model(self):
        return self.model_loaded

    def get_model(self):
        return self.loaded_model

    async def predict(self, files):
        try:
            print("Predicting")
            image = Image.open(files)
            img_tensor = self.transform(image)
            if img_tensor.shape[0] == 4:
                r, g, b, a = img_tensor[0], img_tensor[1], img_tensor[2], img_tensor[3]
            else:
                r, g, b = img_tensor[0], img_tensor[1], img_tensor[2]

            print(img_tensor.shape)

            new_img_tensor = torch.stack([r, g, b], dim=0)
            new_img_np = new_img_tensor.mul(255).byte().numpy()
            new_img_np = np.transpose(new_img_np, (1, 2, 0))
            new_img = Image.fromarray(new_img_np)
            input_tensor = self.transform(new_img)

            with torch.no_grad():
                prediction = self.loaded_model(input_tensor.unsqueeze(0))
                print(prediction)
            predicted_class = torch.argmax(prediction.logits)

            categories = ['airplane', 'automobile', 'bird', 'cat',
                          'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

            return ["success", categories[predicted_class]]

        except Exception as e:
            print(f"Error: {e}")
            return ["Error", str(e)]
