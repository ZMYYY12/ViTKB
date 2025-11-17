import torch
import torch.nn as nn

class KANLayer(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.scale_noise = scale_noise
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.grid_range = grid_range

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )

        self.reset_parameters()

    def reset_parameters(self):
        grid_points = torch.linspace(-1, 1, steps=self.grid_size)
        grid_points = grid_points.unsqueeze(0).expand(self.in_features, -1)
        self.register_buffer('grid_points', grid_points)

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        if y.dim() == 1:
            y = y.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if y.size(1) != self.in_features:
            y = y.T

        if x.size(0) != self.out_features:
            x = x.expand(self.out_features, -1, -1)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)

        solution = torch.linalg.lstsq(A, B.unsqueeze(-1)).solution.squeeze(-1)
        return solution.permute(1, 0)

    def b_splines(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3, f"Input must be 3D, got {x.dim()}D"
        x = x.transpose(1, 2)

        grid = self.grid_points.unsqueeze(0)

        k = self.spline_order
        diff = grid[:, :, k:] - grid[:, :, :-k]
        diff[diff == 0] = 1e-6

        b = torch.zeros_like(x)
        for i in range(k + 1):
            mask = (grid[:, :, i:-1] <= x) & (x < grid[:, :, i + 1:])
            b[mask] = 1

        for d in range(1, k + 1):
            term1 = (x - grid[:, :, : -d]) / (grid[:, :, d:-1] - grid[:, :, : -d] + 1e-6)
            term2 = (grid[:, :, d + 1:] - x) / (grid[:, :, d + 1:] - grid[:, :, 1: -d] + 1e-6)
            b = term1 * b[:, :, : -1] + term2 * b[:, :, 1:]

        return b.transpose(1, 2)

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features, f"Expected x to have shape (batch_size, {self.in_features}), got {x.shape}"
        base_output = torch.matmul(self.base_activation(x), self.base_weight.T)
        spline_output = torch.matmul(self.b_splines(x.view(x.size(0), -1, self.in_features)), self.scaled_spline_weight.view(self.out_features, -1).T)
        return base_output + spline_output

    def update_grid(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.dim() == 3, f"Input must be 3D, got {x.dim()}D"
        self.b_splines(x)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden must have at least two elements")
        for i in range(len(layers_hidden) - 1):
            layer = KANLayer(
                in_features=layers_hidden[i],
                out_features=layers_hidden[i + 1],
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            with torch.no_grad():
                if i > 0:
                    dummy_input = torch.randn(1, grid_size, layers_hidden[i])
                    layer.update_grid(dummy_input)
            self.layers.append(layer)