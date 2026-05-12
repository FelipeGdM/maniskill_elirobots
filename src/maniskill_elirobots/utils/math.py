import numpy as np
import torch
import torch.nn.functional as functional
from quatorch import Quaternion
from torch._tensor import Tensor
from transforms3d.euler import euler2quat


def coin_angle(q: torch.Tensor, normal_axis: torch.Tensor, desired_axis: torch.Tensor) -> torch.Tensor:
    """Computes the angular distance of a coin's face in relation to a desired orientation

    Args:
        q (torch.Tensor): Current coin orientation
        normal_axis (torch.Tensor): Direction normal to the coin's face, in the coin's reference frame
        desired_axis (torch.Tensor): Desired direction for the current coin's face normal vector, in world's reference frame

    Returns:
        angle: Angular distance in degrees
    """
    # num_envs = q.shape[0]

    qnorm = Quaternion(q / torch.linalg.norm(q, axis=1, keepdim=True))

    rotated_vector = qnorm.rotate_vector(normal_axis)

    return torch.acos(functional.cosine_similarity(rotated_vector, desired_axis, dim=1)) * 180 / torch.pi

    # axis, _ = qnorm.to_axis_angle()

    # axis_projection = torch.linalg.vecdot(axis, ref_axis)

    # twist_angle = 2 * torch.atan2(qnorm[..., 0], axis_projection)

    # print(f"{twist_angle.shape=}")

    # qtwist = Quaternion(
    #     torch.stack(
    #         [
    #             torch.cos(twist_angle / 2),
    #             torch.sin(twist_angle / 2) * axis[..., 0],
    #             torch.sin(twist_angle / 2) * axis[..., 1],
    #             torch.sin(twist_angle / 2) * axis[..., 2],
    #         ],
    #         dim=1,
    #     ),
    # )

    # print(f"{qtwist.shape=}")

    # qswing = Quaternion(qnorm * qtwist.conjugate())

    # _, swing_angle = qswing.to_axis_angle()

    # return swing_angle * 180 / torch.pi

    # axis = torch.stack([torch.sin(angle_axis), torch.cos(angle_axis), torch.ones((env_count, 1))], dim=1).reshape((env_count, 3))

    # raw_angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0)) * 180 / torch.pi

    # return 180 - torch.abs(180 - raw_angle)


if __name__ == "__main__":
    num_envs = 1
    initial_orientation = Quaternion(torch.stack([torch.tensor(euler2quat(0, i * np.pi / 10, 0)) for i in range(10)]).reshape(10, 4))
    desired_orientation = Quaternion(torch.tensor(euler2quat(0, -np.pi / 2, 0)).reshape(1, 4))

    normal_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 3)
    desired_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64).reshape(1, 3)

    test_orientation = Quaternion(
        torch.stack(
            [
                torch.tensor(
                    euler2quat(
                        i * np.pi / 10,
                        0,
                        0,
                    ),
                )
                for i in range(10)
            ],
        ),
    )

    # print(initial_orientation.rotate_vector(axis))

    print(desired_orientation.rotate_vector(normal_axis))
    # print(coin_angle(test_orientation, normal_axis, desired_axis))

    # print(
    #     angle_from_quaternion(
    #         desired_orientation,
    #         axis,
    #     ),
    # )
