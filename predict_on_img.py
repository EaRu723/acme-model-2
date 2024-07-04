"""Class for init the model, preprocess the image and make a prediction."""

from transforms.acne_transforms import AcneTransformsTorch
import torch
from model.resnet50 import resnet50


class ModelInit:
    """Class that initialize the model and make prediction on single raw image."""

    def __init__(self, model_type="model_ld_smoothing", path_checkpoint=None, device="cpu"):
        """Init of the object."""
        self.model_type = model_type
        # Create model
        num_acne_cls = 13 if model_type == "model_ld_smoothing" else 4
        self.model = resnet50(num_acne_cls=num_acne_cls)
        # load checkpoint
        checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # transforms
        self.transform = AcneTransformsTorch(train=False)

    def predict_on_img(self, img):
        """Get prediction for given image."""

        cls_test = torch.tensor([], dtype=torch.int32)
        cnt_test = torch.tensor([], dtype=torch.int32)

        self.model.eval()
        with torch.no_grad():
            cls, cou, cou2cls = self.model(self.transform(img)[None, :, :, :])
            # Convert predictions back to Hayashi scale if needed
            if self.model_type == "model_ld_smoothing":
                cls = torch.stack(
                    (
                        torch.sum(cls[:, :1], 1),
                        torch.sum(cls[:, 1:4], 1),
                        torch.sum(cls[:, 4:10], 1),
                        torch.sum(cls[:, 10:], 1),
                    ),
                    1,
                )
            preds_cls = torch.argmax(0.5 * (cls + cou2cls), dim=1)
            preds_cnt = torch.argmax(cou, dim=1) + torch.tensor(1)
            # accumulate predictions
            cls_test = torch.cat((cls_test, preds_cls))
            cnt_test = torch.cat((cnt_test, preds_cnt))

            return cls_test, cnt_test
