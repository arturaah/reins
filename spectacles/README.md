# R1 hand trajectories on Spectacles

This Lens draws two planned hand paths in AR. It reads trajectories in the R1 `robot_base` frame, detects printed shoulder markers, and converts the paths into the Spectacles world-tracked space. The robot can be powered off: both the included WebSocket server and the Lens have animated mock trajectories. The paths are **visual plans only**; nothing here sends motion commands to the R1.

Open `R1 Hand Path Preview.esproj` in Lens Studio 5.15.4. The scene, script, Internet Module, marker assets, World-tracked camera, and cyan/orange materials are already wired. No manual scene setup is needed.

## Print and place the tags

The Lens now tracks the **bare 36h11 tags**: ID **22** on the robot's left shoulder and ID **23** on its right shoulder, as seen from the robot's perspective. The checked-in reference images are `Print/left-shoulder-id-22.svg` and `Print/right-shoulder-id-23.svg`; print at 100% for a 100 mm square image, including its white border. The tracker is configured for that 100 mm image height. Measure Jone's actual cut-out tags; if their image height differs, change `MarkerHeight` on both `.imgmarker` assets to the measured centimetres. Mount the tags flat on top of the blue shoulder dots, facing **upward**. Their in-plane paper rotation does not matter once both tag centres have been scanned. Optional 140 mm labeled cards are also generated, but the Lens matches only the inner tag image.

These are AprilTag **36h11** images used inside Lens Studio **image markers**. Lens Studio matches each tag as an image; it does not decode an AprilTag ID at runtime. The reference images were generated with OpenCV and independently decoded as IDs 22 and 23. Snap's image-marker tracker recognizes one image at a time; scan **both shoulders in sequence** to establish the robot's orientation. Subsequent single-tag sightings update its position; scan both again if the robot turns.

On detection, a short cue appears above the tag and says `LEFT SHOULDER` or `RIGHT SHOULDER` for four seconds. The sound remains throttled to at most once every two seconds; switching shoulders updates the label immediately. Without USB-C, tag tracking and the local mock still run on the glasses. Once a real trajectory has arrived, the Lens holds its last received path if the feed disconnects instead of replacing it with the mock. `ws://127.0.0.1:8765` is only reachable through the USB tunnel; wireless updates use the Mac's hotspot address instead.

For a quick desk detection test, serve the project directory and open `Print/screen-test.html`. It shows both tags together and has buttons for either tag alone. The robot's right tag appears on your left when you face the screen. The paths use robot-frame positions even during this desk test, so they will not emerge from the printed tags.

## Coordinates and alignment

- Trajectory messages use metres in `robot_base`: **+x forward, +y robot-left, +z up**.
- The rendering basis maps robot `[x,y,z]` metres to `[-y,x,z]` centimetres. For the **upward-facing shoulder tags**, the Lens uses the vector from the right tag to the left tag as robot-left, world up as robot-up, and their cross product as robot-forward. This uses the tag centres, so their printed top edges may face different directions. The previous vertical-tag mapping produced a large orientation error.
- The fixed-base MuJoCo model places the shoulder pitch joints at `[0.0325,±0.0857,0.9865]` m. The visual shoulder shells span approximately `x=-0.004..0.069`, `|y|=0.080..0.177`, `z=0.932..1.025` m. For tags on their top blue dots, the current **unmeasured estimate** is left `[0.03,+0.13,1.03]` m, right `[0.03,-0.13,1.03]` m. The two-tag calibration now compares the detected shoulder-centre span with the estimated 26 cm robot span and scales all robot offsets and paths together. This corrects for cut-out tag image size differing from the asset's 10 cm setting. Measure the physical tag centres and update `tagForwardM`, `tagSideM`, and `tagHeightM` for final alignment.
- Planned paths retain their true robot-frame hand positions. The earlier desk test translated each polyline so its first point emerged from a tag; that behavior caused the misplaced paths on the robot and has been removed. The marker-to-robot offset is separate from the hand trajectory.
- After both tags have been sighted within ten seconds, the Lens computes a robot world anchor from their centres and the estimated tag height. The camera has Device Tracking in World mode, so the paths should remain at the robot while the wearer walks around. The last calibrated pose is retained when a marker leaves view; either tag then corrects translation. This behavior still needs an on-glasses walking test.
- This scene sets `allowTemporaryAnchor` false and hides both lines until the pair is calibrated, avoiding misleading paths at a random initial position.

The offline mock starts at the neutral hand sites from the MuJoCo model, `[0.2909,±0.1386,0.771]` m, and reaches forward and upward. It is only a plausible preview, used until a real trajectory arrives; the actual hand start must come from a plan resolved against the robot's measured joints. Cyan is left; orange is right. `pathRadiusCm` controls tube thickness.

## Trajectory feed

On this Mac, run from the project directory:

```sh
python3 trajectory_server.py --host 0.0.0.0 --port 8765
```

The server requires Python 3.10+ and the `websockets` package (already installed on this Mac). It emits four updates per second. This checkout first tries `ws://172.20.10.8:8765` (this Mac's current iPhone-hotspot address), then `ws://127.0.0.1:8765` through an optional **wired ADB reverse tunnel**. The hotspot address may change when devices reconnect: check `ipconfig getifaddr en0`, update `websocketUrl` in `Assets/Scene.scene` or the script Inspector, and resend the Lens. Both the Mac and Spectacles must be on the same hotspot for wireless updates. With the Spectacles attached by USB, `adb reverse tcp:8765 tcp:8765` can provide a fallback, but is not needed for the direct hotspot route. The Lens uses its built-in animated mock until a feed arrives, then holds the last received path if both routes disconnect. On-device access to local `ws://` requires the project's enabled Experimental APIs flag; Lens Studio's ordinary desktop preview does not expose this WebSocket API.

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

To see what the robot would actually do, run `tools/arm_lift.py IFACE --plan ...` without `--execute`. That dry run only subscribes and publishes nothing. It writes the resolved plan, starting from the measured pose and including the measured angles of the joints that stay put, to `sim/plans/arm_lift_dryrun.json`. Serve that file with `plan_feed.py`. The feed re-reads the file when it changes, so the next dry run appears on the glasses without a restart. Stop `trajectory_server.py` first and keep the `adb reverse` tunnel. For reference, the model's shoulder pitch joints sit at `[0.032, ±0.086, 0.986]` m.

### Live remaining path from measured joints

On Artur's Mac, which has the robot Ethernet connection, `tools/relay.py en6` subscribes to `rt/lowstate` and `rt/arm_sdk` and serves read-only joint snapshots on port 8766. This subscription is separate from `tools/arm_lift.py --execute`, which sends commands. On the Mac serving the Lens, run:

```sh
.venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json --state-url ws://ARTUR_MAC_IP:8766
```

Use the same approved plan file for the feed and the command sender. If the feed runs on Artur's Mac, use `ws://127.0.0.1:8766`; if it runs on this Mac, both Macs need a network path to port 8766. The Lens reads the feed on port 8765 (directly over Wi-Fi, or through the USB `adb reverse` tunnel). It never reads DDS or controls the R1.

When the relay reports active `rt/arm_sdk` commands, the feed matches **measured** joint angles against the plan's joint samples, advances monotonically, computes the current hand locations by forward kinematics, and sends only the remaining hand paths. It sends an empty path for a completed or stationary hand, which hides that line. New plan files reset progress. If state or the network goes stale, the last remaining path is held; it is not advanced using elapsed time or anything seen by Spectacles. Joint-space matching can be ambiguous when a plan revisits the same pose, so an explicit execution progress signal would be needed for those plans.

The path starts at the reported hand, not the tag. The tag-to-robot offsets are still estimates until the papers are measured on the robot; set `tagForwardM`, `tagSideM`, and `tagHeightM` to those measurements before evaluating physical alignment.

Each visible hand needs 2–512 finite `[x,y,z]` points; an empty array hides a completed hand. A real R1 planner can send the same schema; the Lens does not need the robot online to test alignment. Reins' `contract/` uses `plan_proposed` messages, so a bridge must extract hand paths and convert them to the robot frame before using the production core.

## Device status and first walking test

Lens Studio imports this scene and runs the renderer without script errors in desktop preview. The Mac's ADB sees the wired Spectacles, and Lens Studio reports a successful wired connection. The ID 22/23 bare-tag version was sent and successfully started on Spectacles; the Lens received the live saved-plan trajectory over USB. Jonathan confirmed that **both physical tags were detected on Spectacles**. A lost tag holds the last world pose. World stability, physical tag dimensions, and assumed shoulder offsets still need testing after mounting the tags on the robot.

Use Lens Studio's **Preview Lens → Send to Spectacles**, then scan the left and right robot-mounted tags within ten seconds. The cyan/orange paths should begin near the model's hand sites, below and forward of the tags, and remain fixed around the robot as you walk. Cover both tags, walk a short distance, then expose one again to observe translation correction; scan both again if the robot turns. When USB-C is removed, the Lens keeps drawing its last received trajectory and world anchor. For continuing joint-position and trajectory updates, the wireless WebSocket route must be reachable; otherwise the displayed plan stays frozen. Restarting the Lens without a feed starts the local mock again.

Snap references: [marker tracking](https://developers.snap.com/lens-studio/features/ar-tracking/world/marker-tracking), [world tracking](https://developers.snap.com/lens-studio/features/ar-tracking/world/tracking-modes), [WebSocket API](https://developers.snap.com/spectacles/about-spectacles-features/apis/web-socket), [connecting Spectacles](https://developers.snap.com/spectacles/get-started/start-building/connecting-lens-studio-to-spectacles), [Unitree R1 dimensions](https://www.unitree.com/mobile/R1/).

The status cue uses a Canvas with ScreenTransform/Text, as in Snap's [Canvas guide](https://developers.snap.com/lens-studio/lens-studio-workflow/scene-set-up/2d/canvas-component). Its placement uses the camera's `back` vector because Snap's [Spectacles CameraProvider API](https://developers.snap.com/lens-studio/api/lens-scripting/interfaces/Packages_SpectaclesInteractionKit_Providers_CameraProvider_CameraProvider.html) says this is the direction in front of the Lens Studio camera. Using `forward` had placed the visual behind the wearer.
