
import torch


def tv_2d_l2(img):
    assert len(img.shape) == 2
    numel = img.shape[0] * img.shape[1]
    x_variance = torch.sum(torch.pow(img[:, :-1] - img[:, 1:], 2))
    y_variance = torch.sum(torch.pow(img[:-1, :] - img[1:, :], 2))
    return (x_variance + y_variance) / (numel)


def tvd_2d_l2(img, direction='x'):
    s = 0
    s += torch.sum(torch.pow(img[:, :-1] - img[:, 1:], 2))
    s += torch.sum(torch.pow(img[:-1, :] - img[1:, :], 2))
    if direction == 'y':
        s += torch.sum(torch.pow(img[:-2, :] - img[2:, :], 2))
    else:
        s += torch.sum(torch.pow(img[:, :-2] - img[:, 2:], 2))
    return s


def tv_2d_l1(img):
    x_variance = torch.sum(torch.abs(img[:, :-1] - img[:, 1:]))
    y_variance = torch.sum(torch.abs(img[:-1, :] - img[1:, :]))
    return (x_variance + y_variance)


def tv_2d_lp(img, p=2):
    x_variance = torch.sum(torch.pow(torch.abs(img[:, :-1] - img[:, 1:]), p))
    y_variance = torch.sum(torch.pow(torch.abs(img[:-1, :] - img[1:, :]), p))
    return (x_variance + y_variance)


def tv_3d_l2(img):
    z_variance = torch.sum(torch.pow(img[:, :, :-1] - img[:, :, 1:], 2))
    y_variance = torch.sum(torch.pow(img[:, :-1, :] - img[:, 1:, :], 2))
    x_variance = torch.sum(torch.pow(img[:-1, :, :] - img[1:, :, :], 2))
    return (x_variance + y_variance + z_variance) / (3 * 512 * 512)


def tv_3d_l1(img):
    z_variance = torch.sum(torch.abs(img[:, :, :-1] - img[:, :, 1:]))
    y_variance = torch.sum(torch.abs(img[:, :-1, :] - img[:, 1:, :]))
    x_variance = torch.sum(torch.abs(img[:-1, :, :] - img[1:, :, :]))
    return (x_variance + y_variance + z_variance)


def tv_3d_lp(img, p=2):
    z_variance = torch.sum(
        torch.pow(torch.abs(img[:, :, :-1] - img[:, :, 1:]), p))
    y_variance = torch.sum(
        torch.pow(torch.abs(img[:, :-1, :] - img[:, 1:, :]), p))
    x_variance = torch.sum(
        torch.pow(torch.abs(img[:-1, :, :] - img[1:, :, :]), p))
    return (x_variance + y_variance + z_variance)
