# Vendored Assets

The robot and gripper models in this directory are vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache License 2.0).

| Subtree | Source | License |
|---|---|---|
| `ur5e/` | `universal_robots_ur5e/` | `ur5e/LICENSE` |
| `gripper/` | `umi_gripper/` | `gripper/LICENSE` |

`scene.xml` is project-original and composes these vendored models.

## Local modifications

The vendored XML files have one small edit so they resolve mesh paths correctly when included from `scene.xml`:

- `ur5e/ur5e.xml`: `meshdir="assets"` → `meshdir="ur5e/assets"`
- `gripper/umi_gripper.xml`: `meshdir="assets" texturedir="assets"` → `meshdir="gripper/assets" texturedir="gripper/assets"`

Loading the vendored XMLs directly (without going through `scene.xml`) will therefore fail to find their meshes. Always load via `scene.xml`.
