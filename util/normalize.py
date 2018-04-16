def norm_gan_to_cls(x):
    return normalize(x, from_range=[-1.0, 1.0], to_range=[0.0, 1.0])


def norm_cls_to_gan(x):
    return normalize(x, from_range=[0.0, 1.0], to_range=[-1.0, 1.0])


def norm_cls_to_vis(x):
    return normalize(x, from_range=[0.0, 1.0], to_range=[0.0, 255.0])


def normalize(x, from_range, to_range):
    # You can pass both np.array and torch.Tensor
    assert isinstance(from_range, list) and isinstance(to_range, list)
    assert len(from_range) == 2 and len(to_range) == 2
    from_min, from_max = from_range
    to_min, to_max = to_range
    assert from_max > from_min and to_max > to_min
    unit_X = (x - from_min) / (from_max - from_min)  # 0 <= unit_X <= 1
    return to_min + (to_max - to_min) * unit_X
