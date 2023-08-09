from compressai.models.waseda import Cheng2020Anchor
from compressai.zoo import cheng2020_anchor


def test_cheng2020_anchor():
    net = cheng2020_anchor(quality=1, pretrained=True)
    Cheng2020Anchor.from_state_dict(net.state_dict())


if __name__ == '__main__':
    test_cheng2020_anchor()