# R1 hand trajectories on Spectacles

This Lens draws two planned hand paths in AR. It reads trajectories in the R1 `robot_base` frame, detects printed shoulder markers, and converts the paths into the Spectacles world-tracked space. The robot can be powered off: both the included WebSocket server and the Lens have animated mock trajectories. The paths are **visual plans only**; nothing here sends motion commands to the R1.

Open `R1 Hand Path Preview.esproj` in Lens Studio 5.15.4. The scene, script, Internet Module, marker assets, World-tracked camera, and cyan/orange materials are already wired. No manual scene setup is needed.

## Print and place the tags

The Lens now tracks the **bare 36h11 tags**: ID **22** on the robot's left shoulder and ID **23** on its right shoulder, as seen from the robot's perspective. The checked-in reference images are `Print/left-shoulder-id-22.svg` and `Print/right-shoulder-id-23.svg`; print at 100% for a 100 mm square image, including its white border. The tracker is configured for that 100 mm image height. Measure Jone's actual cut-out tags; if their image height differs, change `MarkerHeight` on both `.imgmarker` assets to the measured centimetres. Mount the tags on flat, forward-facing, roughly vertical surfaces. Optional 140 mm labeled cards are also generated, but the Lens matches only the inner tag image.

These are AprilTag **36h11** images used inside Lens Studio **image markers**. Lens Studio matches each tag as an image; it does not decode an AprilTag ID at runtime. The reference images were generated with OpenCV and independently decoded as IDs 22 and 23. Snap's image-marker tracker recognizes one image at a time; either visible shoulder is enough for this implementation.

On detection, a short cue appears above the tag and says `LEFT SHOULDER` or `RIGHT SHOULDER` for four seconds. The sound remains throttled to at most once every two seconds; switching shoulders updates the label immediately. Without USB-C, tag tracking and the local mock still run on the glasses, but `ws://127.0.0.1:8765` is no longer reachable, so the Mac's saved-plan or live trajectory feed stops updating.

For a quick desk test, serve the project directory and open `Print/screen-test.html`. It shows both cards together and has buttons for either card alone. The robot's right card appears on your left when you face the screen. If the screen cannot fit two 14 cm cards, the page reduces them to fit. The default preview half-spacing of 7.1 cm assumes adjacent nominal 14 cm cards with a 2 mm gap; change `previewTagHalfSpacingM` if the displayed or printed centres differ. Screen scaling still makes metric depth approximate.

## Coordinates and alignment

- Trajectory messages use metres in `robot_base`: **+x forward, +y robot-left, +z up**.
- The marker's +x is image-right, +y image-up, and +z points out of its front. For forward-facing upright tags, the script maps robot `[x,y,z]` in metres to marker `[y,z,x]` in centimetres.
- Assumed marker centres in `robot_base`: left `[0.04,+0.16,1.02]` m; right `[0.04,-0.16,1.02]` m. These are estimates based on the R1's approximately 1.23 m standing height, **not measured shoulder offsets**. Change `tagForwardM`, `tagSideM`, and `tagHeightM` in the controller's script inputs after measuring the mounted tags.
- `previewFromTags` is enabled for the desk test: each polyline is translated so its first point emerges from its corresponding tag centre. This preserves the trajectory shape but deliberately overrides the hand's robot-frame start position. Once the tags are on the R1 and their centres are measured, set `previewFromTags` false and use the measured marker offsets to render the original hand positions.
- On detection, the Lens computes `T_world_robot = T_world_tag × inverse(T_robot_tag)` and places both paths under that world-space robot anchor. The camera has Device Tracking in World mode, so the paths should remain at the robot while the wearer walks around. The last calibrated pose is retained when a marker leaves view; sighting a tag again corrects accumulated drift. This behavior still needs an on-glasses walking test.
- Before the first tag sighting, a temporary mock anchor appears about 1.5 m ahead of the wearer. It is replaced by the tag-based anchor as soon as either tag is recognized. Set `allowTemporaryAnchor` false to show paths only after marker detection.

The mock path starts around `[0.08,±0.23,0.68]` m in the message and arcs forward toward `x≈0.50` m. In preview mode, its first rendered point is moved to its tag. Cyan is left; orange is right. `pathRadiusCm` controls tube thickness.

## Trajectory feed

On this Mac, run from the project directory:

```sh
python3 trajectory_server.py --host 0.0.0.0 --port 8765
```

The server requires Python 3.10+ and the `websockets` package (already installed on this Mac). It emits four updates per second. The script currently points at `ws://127.0.0.1:8765` for a **wired ADB reverse tunnel**. With the Spectacles attached by USB, run `adb reverse tcp:8765 tcp:8765` before previewing the Lens. Run `adb reverse --list` to confirm the tunnel. A wireless setup can instead point `websocketUrl` at the Mac's LAN address (`ipconfig getifaddr en0`), but both devices must be on a network that permits client-to-client traffic. The Lens uses its built-in animated mock if no feed arrives. On-device access to local `ws://` requires the project's enabled Experimental APIs flag; Lens Studio's ordinary desktop preview does not expose this WebSocket API.

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

To see what the robot would actually do, run `tools/arm_lift.py IFACE --plan ...` without `--execute`. That dry run only subscribes and publishes nothing. It writes the resolved plan, starting from the measured pose and including the measured angles of the joints that stay put, to `sim/plans/arm_lift_dryrun.json`. Serve that file with `plan_feed.py`. The feed re-reads the file when it changes, so the next dry run appears on the glasses without a restart. Stop `trajectory_server.py` first and keep the `adb reverse` tunnel. For reference, the model's shoulder pitch joints sit at `[0.032, ±0.086, 0.986]` m.

### Live remaining path from measured joints

On Artur's Mac, which has the robot Ethernet connection, `tools/relay.py en6` subscribes to `rt/lowstate` and `rt/arm_sdk` and serves read-only joint snapshots on port 8766. This subscription is separate from `tools/arm_lift.py --execute`, which sends commands. On the Mac serving the Lens, run:

```sh
.venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json --state-url ws://ARTUR_MAC_IP:8766
```

Use the same approved plan file for the feed and the command sender. If the feed runs on Artur's Mac, use `ws://127.0.0.1:8766`; if it runs on this Mac, both Macs need a network path to port 8766. The Lens reads the feed on port 8765 (directly over Wi-Fi, or through the USB `adb reverse` tunnel). It never reads DDS or controls the R1.

When the relay reports active `rt/arm_sdk` commands, the feed matches **measured** joint angles against the plan's joint samples, advances monotonically, computes the current hand locations by forward kinematics, and sends only the remaining hand paths. It sends an empty path for a completed or stationary hand, which hides that line. New plan files reset progress. If state or the network goes stale, the last remaining path is held; it is not advanced using elapsed time or anything seen by Spectacles. Joint-space matching can be ambiguous when a plan revisits the same pose, so an explicit execution progress signal would be needed for those plans.

The Lens ignores `previewFromTags` for measured-state messages so the path starts at the reported hand, not the tag. The tag-to-robot offsets are still estimates until the cards are mounted and measured; set `tagForwardM`, `tagSideM`, and `tagHeightM` to those measurements before evaluating physical alignment.

Each visible hand needs 2–512 finite `[x,y,z]` points; an empty array hides a completed hand. A real R1 planner can send the same schema; the Lens does not need the robot online to test alignment. Reins' `contract/` uses `plan_proposed` messages, so a bridge must extract hand paths and convert them to the robot frame before using the production core.

## Device status and first walking test

Lens Studio imports this scene and runs the renderer without script errors in desktop preview. The Mac's ADB sees the wired Spectacles, and Lens Studio reports a successful wired connection. The ID 22/23 bare-tag version was sent and successfully started on Spectacles; the Lens received the live saved-plan trajectory over USB. Jonathan confirmed that **both physical tags were detected on Spectacles**. A lost tag holds the last world pose. World stability, physical tag dimensions, and assumed shoulder offsets still need testing after mounting the tags on the robot.

With the USB tunnel active, use Lens Studio's **Preview Lens → Send to Spectacles**. With the robot off, hold a printed tag at roughly shoulder height or mount it on a stand. Move the glasses laterally and around the tag: the cyan/orange paths should remain in the same place. Cover the tag, walk a short distance, then expose it again to observe relocalization. If the paths appear mirrored or point into the robot, check that both tags are upright and forward-facing before adjusting the coordinate mapping. When USB-C is removed, the Lens keeps drawing its local mock trajectory, but the live Mac WebSocket feed via `adb reverse` stops; wireless live trajectories need a reachable Wi-Fi host/port configured in `websocketUrl`.

Snap references: [marker tracking](https://developers.snap.com/lens-studio/features/ar-tracking/world/marker-tracking), [world tracking](https://developers.snap.com/lens-studio/features/ar-tracking/world/tracking-modes), [WebSocket API](https://developers.snap.com/spectacles/about-spectacles-features/apis/web-socket), [connecting Spectacles](https://developers.snap.com/spectacles/get-started/start-building/connecting-lens-studio-to-spectacles), [Unitree R1 dimensions](https://www.unitree.com/mobile/R1/).

The status cue uses a Canvas with ScreenTransform/Text, as in Snap's [Canvas guide](https://developers.snap.com/lens-studio/lens-studio-workflow/scene-set-up/2d/canvas-component). Its placement uses the camera's `back` vector because Snap's [Spectacles CameraProvider API](https://developers.snap.com/lens-studio/api/lens-scripting/interfaces/Packages_SpectaclesInteractionKit_Providers_CameraProvider_CameraProvider.html) says this is the direction in front of the Lens Studio camera. Using `forward` had placed the visual behind the wearer.
