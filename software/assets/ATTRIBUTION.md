# Vendored Assets

The robot and gripper models in this directory are vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache License 2.0).

| Subtree | Source | License |
|---|---|---|
| `ur5e/` | `universal_robots_ur5e/` | `ur5e/LICENSE` |
| `gripper/` | `umi_gripper/` | `gripper/LICENSE` |

`scene.xml` is project-original and composes these vendored models.

## Local modifications

The UR5e XML has one small edit so it resolves mesh paths correctly when included from `scene.xml`:

- `ur5e/ur5e.xml`: `meshdir="assets"` → `meshdir="ur5e/assets"`

The gripper is attached as an MJCF sub-model from `scene.xml`, so its mesh and texture paths stay relative to
`gripper/umi_gripper.xml`: `meshdir="assets" texturedir="assets"`.
