"""WeightEMA: shadow == hand-computed EMA of every floating state entry, integer buffers copied,
state_dict loadable into a fresh module, reset/load round-trips. CPU only."""
import copy

import torch

from src.utils.weight_ema import WeightEMA


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 2))


def test_matches_hand_computed_average():
    net = _net()
    ema = WeightEMA(net, decay=0.9)
    ref = {k: v.detach().float().clone() for k, v in net.state_dict().items() if v.is_floating_point()}
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for n in range(1, 21):
        loss = net(torch.randn(16, 4)).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update()
        d = min(0.9, (1 + n) / (10 + n))
        for k, v in net.state_dict().items():
            if v.is_floating_point():
                ref[k] = d * ref[k] + (1 - d) * v.detach().float()
    for k, v in ref.items():
        assert torch.allclose(ema.shadow[k], v, atol=1e-6), k
    # integer buffer copied, not averaged
    assert int(ema.shadow['1.num_batches_tracked']) == int(net.state_dict()['1.num_batches_tracked']) == 20


def test_state_dict_loads_and_round_trips():
    net = _net()
    ema = WeightEMA(net, decay=0.99)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    for _ in range(5):
        net(torch.randn(16, 4)).pow(2).mean().backward(); opt.step(); opt.zero_grad(); ema.update()
    fresh = _net()
    fresh.load_state_dict(ema.state_dict(), strict=True)       # keys + dtypes compatible
    other = WeightEMA(copy.deepcopy(net), decay=0.99)
    other.load_state_dict(ema.state_dict(), num_updates=ema.num_updates)
    for k in ema.shadow:
        assert torch.equal(other.shadow[k], ema.shadow[k]), k
    assert other.num_updates == ema.num_updates == 5


def test_reset_and_identity_when_weights_constant():
    net = _net()
    ema = WeightEMA(net, decay=0.999)
    for _ in range(3):
        ema.update()                                           # weights unchanged -> average unchanged
    for k, v in net.state_dict().items():                      # up to float rounding of d*x + (1-d)*x
        assert torch.allclose(ema.state_dict()[k].float(), v.float(), rtol=1e-6, atol=1e-7), k
    with torch.no_grad():
        net[0].weight.add_(1.0)
    ema.reset()
    assert torch.equal(ema.state_dict()['0.weight'], net.state_dict()['0.weight'])
    assert ema.num_updates == 0


if __name__ == '__main__':
    test_matches_hand_computed_average(); test_state_dict_loads_and_round_trips(); test_reset_and_identity_when_weights_constant()
    print('weight_ema tests: 3 passed')
