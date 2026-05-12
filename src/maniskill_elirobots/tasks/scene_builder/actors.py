import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.building.actors.common import _build_by_type  # pyright: ignore[reportPrivateUsage]
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


def build_twocolor_cylinder(  # noqa: PLR0913
    scene: ManiSkillScene,
    radius: float,
    half_length: float,
    color_1: list[float],
    color_2: list[float],
    name: str,
    *,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Array | None = None,
    initial_pose: Pose | sapien.Pose | None = None,
):

    builder: ActorBuilder = scene.create_actor_builder()

    if add_collision:
        _ = builder.add_cylinder_collision(
            radius=radius,
            half_length=half_length,
        )

    _ = builder.add_cylinder_visual(
        pose=sapien.Pose(p=[-half_length / 2, 0, 0]),
        radius=radius,
        half_length=half_length / 2,
        material=sapien.render.RenderMaterial(
            base_color=color_1,
        ),
    )

    _ = builder.add_cylinder_visual(
        pose=sapien.Pose(p=[half_length / 2, 0, 0]),
        radius=radius,
        half_length=half_length / 2,
        material=sapien.render.RenderMaterial(
            base_color=color_2,
        ),
    )

    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)


def build_transparent_sphere(  # noqa: PLR0913
    scene: ManiSkillScene,
    radius: float,
    color,
    name: str,
    *,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Array | None = None,
    initial_pose: Pose | sapien.Pose | None = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        _ = builder.add_sphere_collision(
            radius=radius,
        )
    _ = builder.add_sphere_visual(
        radius=radius,
        material=sapien.render.RenderMaterial(
            base_color=color,
            transmission=0.75,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)
