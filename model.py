import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, D: int, W: int, input_ch: int, input_ch_d: int, skips=[4]):
        super(NeRF, self).__init__()
        """
        D : Layers in Network (8)  ||  W : Channels per Layer (256)
        input_ch : input from pos_enc (x,y,z)  ||  input_ch_d : input from pos_enc (d)
        output_ch : 5 ?   ||   skips : [4] ? 
        """
        self.D = D
        self.W = W
        self.input_ch_x = input_ch
        self.input_ch_d = input_ch_d
        self.skips = skips

        self.linear_x = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.linear_d = nn.Linear(input_ch_d + W, W//2)

        self.linear_feat = nn.Linear(W, W)
        self.linear_density = nn.Linear(W, 1)
        self.linear_color = nn.Linear(W//2, 3)
        # self.linear_output = nn.Linear(W, output_ch)    # Input 값에 Direction 안 쓸때 사용 (5D -> 3D)

    def forward(self, x):
        input_x, input_d = torch.split(x, [self.input_ch_x, self.input_ch_d], dim=-1)
        out = input_x
        # [0~7] for 8 Layers
        for i, l in enumerate(self.linear_x):
            out = self.linear_x[i](out)
            out = F.relu(out)
            if i in self.skips:
                out = torch.cat([input_x, out], dim=-1)
        # [8-1], [8-2]
        density = self.linear_density(out)
        feature = self.linear_feat(out)
        # [9]
        out = torch.cat([feature, input_d], dim=-1)
        out = self.linear_d(out)
        out = F.relu(out)
        # [10]
        out = self.linear_color(out)
        result = torch.cat([out, density], dim=-1)
        return result


class NeRFs(nn.Module):
    def __init__(self,  D: int, W: int, input_ch: int, input_ch_d: int, skips=[4]):
        # (D=8, W=256, input_ch=63, input_ch_d=27, skips=[4])
        super().__init__()
        self.nerf_c = NeRF(D, W, input_ch, input_ch_d, skips)
        self.nerf_f = NeRF(D, W, input_ch, input_ch_d, skips)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, x, is_fine=False):
        if is_fine:
            return self.nerf_f(x)
        return self.nerf_c(x)


if __name__ == "__main__":
    # device_ids = [0]
    # device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')
    model = NeRFs(D=8, W=256, input_ch=63, input_ch_d=27, skips=[4])# .to(device)
    input_test = torch.rand(65536, 90) # .to(device)       # torch.Size([65536, 63+27])
    result_test = model(input_test, is_fine=True)
    print(result_test.shape)
    print(model.parameters())
