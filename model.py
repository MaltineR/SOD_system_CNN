import torch
import torch.nn as nn


class BaselineSODModel(nn.Module):
    def __init__(self):
        super(BaselineSODModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 -> 64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 16 -> 8
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 8 -> 16
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 16 -> 32
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),   # 32 -> 64
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),    # 64 -> 128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Test model
if __name__ == "__main__":
    model = BaselineSODModel()

    test_input = torch.randn(8, 3, 128, 128)
    output = model(test_input)

    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)

    if output.shape == torch.Size([8, 1, 128, 128]):
        print("Model test successful.")
    else:
        print("Model output shape is wrong.")