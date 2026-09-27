# R1 hand trajectories on Spectacles

This Lens draws two planned hand paths in AR. It reads trajectories in the R1 `robot_base` frame or an initial-pose `map` frame, detects printed shoulder markers, and converts the paths into the Spectacles world-tracked space. The robot can be powered off: both the included WebSocket server and the Lens have animated mock trajectories. The paths are **visual plans only**; nothing here sends motion commands to the R1.

Open `R1 Hand Path Preview.esproj` in Lens Studio 5.15.4. The scene, script, Internet Module, marker assets, World-tracked camera, and cyan/orange materials are already wired. No manual scene setup is needed.

## Print and place the tags

The Lens now tracks the **bare 36h11 tags**: ID **22** on the robot's left shoulder and ID **23** on its right shoulder, as seen from the robot's perspective. The checked-in reference images are `Print/left-shoulder-id-22.svg` and `Print/right-shoulder-id-23.svg`. The robot's cut-out tags have approximately **4 cm black squares**. In the tracked PNG asset, the black square occupies 80% of the full image height, so both `.imgmarker` assets are now configured for a **5 cm full image height**. If the cut-outs are measured more precisely, set `MarkerHeight` to the full image height in centimetres (black-square width divided by 0.8). Mount the tags flat on top of the blue shoulder dots, facing **upward**. Their in-plane paper rotation does not matter once both tag centres have been scanned. The optional 100 mm printable image and 140 mm labeled card are larger reference prints; the robot cut-outs are smaller.

These are AprilTag **36h11** images used inside Lens Studio **image markers**. Lens Studio matches each tag as an image; it does not decode an AprilTag ID at runtime. The reference images were generated with OpenCV and independently decoded as IDs 22 and 23. Snap's image-marker tracker recognizes one image at a time; scan **both shoulders in sequence** to establish the robot's orientation. Subsequent single-tag sightings update its position; scan both again if the robot turns.

On detection, a short cue appears above the tag and says `LEFT SHOULDER` or `RIGHT SHOULDER` for four seconds. The sound remains throttled to at most once every two seconds; switching shoulders updates the label immediately. Without USB-C, tag tracking and the local mock still run on the glasses. Once a real trajectory has arrived, the Lens holds its last received path if the feed disconnects instead of replacing it with the mock. `ws://127.0.0.1:8765` is only reachable through the USB tunnel; wireless updates use the Mac's current LAN address (`192.168.1.145` on the hackathon Wi-Fi). Update `websocketUrl` in the scene when that address changes.

For a quick desk detection test, serve the project directory and open `Print/screen-test.html`. It shows both tags together and has buttons for either tag alone. The robot's right tag appears on your left when you face the screen. The paths use robot-frame positions even during this desk test, so they will not emerge from the printed tags.

## Coordinates and alignment

- Trajectory messages use metres in `robot_base`: **+x forward, +y robot-left, +z up**.
- The rendering basis maps robot `[x,y,z]` metres to `[-y,x,z]` centimetres. For the **upward-facing shoulder tags**, the Lens uses the vector from the right tag to the left tag as robot-left, world up as robot-up, and their cross product as robot-forward. This uses the tag centres, so their printed top edges may face different directions. The previous vertical-tag mapping produced a large orientation error.
- The fixed-base MuJoCo model places the shoulder pitch joints at `[0.0325,±0.0857,0.9865]` m. The visual shoulder shells span approximately `x=-0.004..0.069`, `|y|=0.080..0.177`, `z=0.932..1.025` m. The photos place the stickers on the dark top shoulder caps. We added approximate sticker squares and `left_shoulder_tag_center` / `right_shoulder_tag_center` sites to `sim/models/r1/R1_fixed_base.xml`. In the model's zero-joint pose, their `robot_base` coordinates are `[0.030,+0.130,1.019]` m and `[0.030,-0.130,1.019]` m. This is a model-based estimate, not a measured survey of the robot. The neutral hand sites are `[0.2909,±0.1386,0.7710]` m, so each path starts about **26.1 cm forward and 24.8 cm below** its tag in the zero-joint model. The Lens uses these tag coordinates. The 5 cm marker-image size and 26 cm tag-centre span are still approximate; two-tag calibration applies a residual scale correction when they differ.
- The stickers sit on the shoulder-pitch links, which are movable. Their `robot_base` coordinates above apply to the model's zero-joint pose. The sites follow the links in MuJoCo; after robot joint telemetry is connected, the calibration should use the measured shoulder and waist joints when the arms or torso move. Until then, rescan both tags in a similar arm pose, or treat a moving-arm calibration as approximate.
- Planned paths retain their true robot-frame hand positions. The earlier desk test translated each polyline so its first point emerged from a tag; that behavior caused the misplaced paths on the robot and has been removed. The marker-to-robot offset is separate from the hand trajectory.
- After both tags have been sighted within ten seconds, the Lens computes a robot world anchor from their centres and the estimated tag height. The camera has Device Tracking in World mode, so the paths should remain at the robot while the wearer walks around. The last calibrated pose is retained when a marker leaves view; either tag then corrects translation. This behavior still needs an on-glasses walking test.
- This scene sets `allowTemporaryAnchor` false and hides both lines until the pair is calibrated, avoiding misleading paths at a random initial position.

The offline mock starts at the neutral hand sites from the MuJoCo model, `[0.2909,±0.1386,0.771]` m, and reaches forward and upward. It is only a plausible preview, used until a real trajectory arrives; the actual hand start must come from a plan resolved against the robot's measured joints. Cyan is left; orange is right. `pathRadiusCm` controls tube thickness.

## Trajectory feed

On this Mac, run from the project directory:

```sh
python3 trajectory_server.py --host 0.0.0.0 --port 8765
```

For an alignment check with fixed paths, add `--static`. This holds both
paths at the model's neutral hand starts and leaves their endpoints unchanged;
it does not represent measured robot motion.

The server requires Python 3.10+ and the `websockets` package. It emits four updates per second. The current Lens tries Jonathan's hackathon-Wi-Fi address (`ws://192.168.1.145:8765`), then `ws://127.0.0.1:8765` through a **wired ADB reverse tunnel**. That first address is only a snapshot: it will not point to Artur's Mac. For a wired demo on Artur's Mac, plug the glasses into his Mac and run `adb reverse tcp:8765 tcp:8765`; the fallback URL then reaches his local feed without editing or rebuilding the Lens. To try wireless later, put Artur's Mac and the glasses on the same hotspot, set `websocketUrl` in the Lens Inspector to Artur's hotspot IP, save, and resend the Lens. The Lens uses its built-in animated mock until a feed arrives, then holds the last received path if both routes disconnect. On-device access to local `ws://` requires the project's enabled Experimental APIs flag; Lens Studio's ordinary desktop preview does not expose this WebSocket API.

The included server is an **animated trajectory test**, not robot telemetry. Both hands start at the model's neutral hand sites; their planned endpoints change shape roughly every nine seconds while their starting points stay fixed. It can verify that the Lens receives live plan updates over Wi-Fi, but it cannot show measured hand movement or shorten a path as the robot moves.

Example message:

```json
{
  "type": "trajectory", "version": 1, "id": "plan-1",
  "frame": "robot_base", "units": "m",
  "hands": {
    "left": [[0.08, 0.23, 0.68], [0.50, 0.33, 0.90]],
    "right": [[0.08, -0.23, 0.72], [0.50, -0.33, 0.94]]
  }
}
```

### Real plans instead of the mock

`plan_feed.py` serves any Reins plan file (schema_version 1, MuJoCo joint names: `sim/plans/`, `tools/plans/`, `recordings/`) in the same message format, on the same port. It poses the fixed-base R1 model along the plan (forward kinematics, no physics) and sends both hand-tip paths in `robot_base`, which is the model's world frame (pelvis pinned 0.74 m above the floor origin). The plan feed never commands the robot. From the repository root:

```sh
.venv/bin/python spectacles/plan_feed.py tools/plans/cup_grab_right.json
.venv/bin/python spectacles/plan_feed.py tools/plans/cup_grab_right.json --print   # one message, no server
```

### One-Mac wired demo on Artur's robot Mac

Artur does **not** need Lens Studio to serve trajectories or read robot joints. The already-sent Lens can use USB to his Mac while it remains running on the glasses. Lens Studio is needed only to install, resend, or edit the Lens. The robot stays on its Ethernet link; the Mac's Wi-Fi or hotspot is independent of the wired USB feed.

From Artur's copy of this repository, with the R1 on its existing Ethernet interface (replace `en6` if his interface differs):

```sh
# Once per USB connection, with the glasses plugged into Artur's Mac:
adb devices
adb reverse tcp:8765 tcp:8765

# Terminal 1: read robot pose and serve remaining hand paths; no robot commands:
.venv/bin/python spectacles/plan_feed.py tools/plans/cup_grab_right.json --robot-iface en6
```

`adb` is supplied by Homebrew's `android-platform-tools` if it is missing. The Python environment needs `mujoco`, `numpy`, `websockets`, and `unitree_sdk2py` as used by the existing robot tools. The feed prints `listening on en6, domain 0` when DDS starts, `loaded ...` when the plan is valid, and `Lens connected: ...` when the glasses reach it. The `rt/lowstate` readings must arrive before the path can shorten; a plan file can still be shown without robot state. The current plan is re-read whenever its file changes.

To start from the robot's actual measured arm pose, first dry-run the plan on Artur's Mac with the existing command tool, then serve its resolved output instead:

```sh
.venv/bin/python tools/arm_lift.py en6 --plan tools/plans/cup_grab_right.json
.venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json --robot-iface en6
```

The dry run and feed above only read the robot; neither publishes a movement command. If a separate controller later sends the corresponding `rt/arm_sdk` command, the feed uses measured `rt/lowstate` joints to shorten the path. If that controller writes a new plan JSON file, the feed reloads it and updates the glasses. There is no need to run `tools/relay.py` when `--robot-iface` is used on the robot-side Mac. `--state-url` remains available when the feed runs on a different computer.

To see what the robot would actually do, run `tools/arm_lift.py IFACE --plan ...` without `--execute`. That dry run only subscribes and publishes nothing. It writes the resolved plan, starting from the measured pose and including the measured angles of the joints that stay put, to `sim/plans/arm_lift_dryrun.json`. Serve that file with `plan_feed.py`. The feed re-reads the file when it changes, so the next dry run appears on the glasses without a restart. Stop `trajectory_server.py` first and keep the `adb reverse` tunnel. For reference, the model's shoulder pitch joints sit at `[0.032, ±0.086, 0.986]` m.

### Live remaining path from measured joints

On Artur's Mac, which has the robot Ethernet connection, `tools/relay.py en6` subscribes to `rt/lowstate` and `rt/arm_sdk` and serves read-only joint snapshots on port 8766. This subscription is separate from `tools/arm_lift.py --execute`, which sends commands. On the Mac serving the Lens, run:

```sh
.venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json --state-url ws://ARTUR_MAC_IP:8766
```

Use the same approved plan file for the feed and the command sender. If the feed runs on Artur's Mac, use `ws://127.0.0.1:8766`; if it runs on this Mac, both Macs need a network path to port 8766. The Lens reads the feed on port 8765 (directly over Wi-Fi, or through the USB `adb reverse` tunnel). It never reads DDS or controls the R1.

When the relay reports active `rt/arm_sdk` commands, the feed matches **measured** joint angles against the plan's joint samples, advances monotonically, computes the current hand locations by forward kinematics, and sends only the remaining hand paths. A completed or stationary hand is shown as a point at its measured position; before execution, moving hands show their full proposed paths. New plan files reset progress. If state or the network goes stale, the last path and hand points are held; they are not advanced using elapsed time or anything seen by Spectacles. Joint-space matching can be ambiguous when a plan revisits the same pose, so an explicit execution progress signal would be needed for those plans.

The path starts at the reported hand, not the tag. The tag-to-robot offsets are still estimates until the papers are measured on the robot; set `tagForwardM`, `tagSideM`, and `tagHeightM` to those measurements before evaluating physical alignment.

### Planned walk followed by an arm action

`sim/preview.py` pins the pelvis and previews arm motion only. For an AR demonstration that includes walking, add `base_keyframes` to a resolved arm plan. These are planned planar robot-base poses in metres and radians. `map` has the same axes as `robot_base` at the **initial** pose: +x forward, +y left, +z up. The feed applies each planned base pose to the MuJoCo hand-tip positions and sends a room-fixed `frame: "map"` trajectory. For a straight 0.7 m walk followed by the dry-run arm action:

```sh
.venv/bin/python spectacles/make_walk_plan.py sim/plans/arm_lift_dryrun.json \
  --distance-m 0.7 --walk-s 4 --output sim/plans/walk_then_arm.json
.venv/bin/python spectacles/preview_walk.py sim/plans/walk_then_arm.json \
  --output outputs/walk-plan.png
.venv/bin/python spectacles/plan_feed.py sim/plans/walk_then_arm.json --robot-iface en8
```

The preview PNG plots the same kinematic hand paths that the feed sends, from above and from the side. It is not a balanced MuJoCo walking simulation. The generator does not command locomotion. Replace 0.7 m and 4 s with Artur's actual planned displacement and duration; arbitrary turns can be described by editing the plan's `base_keyframes` (`time_s`, `x_m`, `y_m`, `yaw_rad`). The final base keyframe must cover the full combined plan. Serve a new plan file whenever the walking or arm plan changes.

Resend the updated Lens once from Jonathan's Mac, which has Lens Studio. With the glasses and Artur's Mac on the same hotspot, the existing Inspector URL `ws://172.20.10.7:8765` points to Artur's feed as long as his hotspot address stays the same. Scan **both tags while the robot is at the plan's starting pose, before it walks**. The Lens then holds the `map` anchor in the room while the robot moves; shoulder tags can still trigger their labels and sound, but a moving shoulder does not pull the future path along. Switching between `robot_base` and `map` plans requires scanning both tags again. The glasses do not estimate robot progress from camera motion.

The current `rt/lowstate` joint feed does not include a measured room pose for the base. Without one, the complete walk-and-arm path stays visible even as the robot moves. To shorten it from **measured** walking and arm motion, Artur's controller can atomically update a JSON file at least once per second:

```json
{"frame":"map","x_m":0.35,"y_m":0.0,"yaw_rad":0.0}
```

The values must be measured relative to the robot's initial base pose, in the same map axes as the plan. Pass its path to `plan_feed.py` with `--base-pose-file /path/to/base_pose.json --robot-iface en8`. The feed also requires the measured joints; it rejects stale base files and never advances from elapsed time. If no measured base pose is available, omit that option and use the full planned path for the visual demo. This does not add walking control or odometry to the robot.

Each hand accepts 1–512 finite `[x,y,z]` points: one point draws a hand marker, two or more draw a path, and an empty array hides it. A real R1 planner can send the same schema; the Lens does not need the robot online to test alignment. Reins' `contract/` uses `plan_proposed` messages, so a bridge must extract hand paths and convert them to the robot frame before using the production core.

## Device status and first walking test

Lens Studio imports this scene and runs the renderer without script errors in desktop preview. The Mac's ADB sees the wired Spectacles, and Lens Studio reports a successful wired connection. The ID 22/23 bare-tag version was sent and successfully started on Spectacles; the Lens received the live saved-plan trajectory over USB. Jonathan confirmed that **both physical tags were detected on Spectacles**. A lost tag holds the last world pose. World stability, physical tag dimensions, and assumed shoulder offsets still need testing after mounting the tags on the robot.

Use Lens Studio's **Preview Lens → Send to Spectacles**, then scan the left and right robot-mounted tags within ten seconds. The cyan/orange paths should begin near the model's hand sites, below and forward of the tags, and remain fixed around the robot as you walk. Cover both tags, walk a short distance, then expose one again to observe translation correction; scan both again if the robot turns. When USB-C is removed, the Lens keeps drawing its last received trajectory and world anchor. For continuing joint-position and trajectory updates, the wireless WebSocket route must be reachable; otherwise the displayed plan stays frozen. Restarting the Lens without a feed starts the local mock again.

Snap references: [marker tracking](https://developers.snap.com/lens-studio/features/ar-tracking/world/marker-tracking), [world tracking](https://developers.snap.com/lens-studio/features/ar-tracking/world/tracking-modes), [WebSocket API](https://developers.snap.com/spectacles/about-spectacles-features/apis/web-socket), [connecting Spectacles](https://developers.snap.com/spectacles/get-started/start-building/connecting-lens-studio-to-spectacles), [Unitree R1 dimensions](https://www.unitree.com/mobile/R1/).

The status cue uses a Canvas with ScreenTransform/Text, as in Snap's [Canvas guide](https://developers.snap.com/lens-studio/lens-studio-workflow/scene-set-up/2d/canvas-component). Its placement uses the camera's `back` vector because Snap's [Spectacles CameraProvider API](https://developers.snap.com/lens-studio/api/lens-scripting/interfaces/Packages_SpectaclesInteractionKit_Providers_CameraProvider_CameraProvider.html) says this is the direction in front of the Lens Studio camera. Using `forward` had placed the visual behind the wearer.
