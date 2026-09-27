# Spectacles review and voice

Open `R1 Hand Path Preview.esproj` in Lens Studio 5.15.4. The scene includes the
trajectory renderer, microphone, shoulder markers and materials. Save and resend
the Lens after changing code or connection settings.

Run the dashboard and open **Connections → Glasses motion review**. Create a
paired device, then copy its `deviceId`, `reviewToken`, and WebSocket URL into
the Lens Inspector. Leave `pairingToken` empty and `demoMode` disabled for this
workflow. The secret is shown once. Pairing survives dashboard restarts and can
be revoked in the same panel. Only hashes are stored in
`$REINS_STATE_DIR/glasses_pairing.json` (default `~/.local/state/reins`), outside
the repository. Do not commit a scene with populated credentials.

For Wi-Fi, set `websocketUrl` to `ws://MAC_WIFI_IP:8765` on the same private
network and clear `fallbackWebsocketUrl`. For USB, run
`adb reverse tcp:8765 tcp:8765` and use `ws://127.0.0.1:8765`. Dashboard pairing
is required in either case. Local `ws://` uses the project's Experimental APIs
setting; it authenticates access but does not encrypt transport.

## Drafts and complete motion review

The agent can display and revise draft paths without asking for approval.
Only a submitted complete proposal shows a review card: double right pinch
approves once; double left pinch rejects. Each decision is bound to the current
connection, proposal revision and exact digest. Reconnecting clears any partial
acceptance gesture. The browser and glasses review the same proposal.

Both shoulder tags must have been seen within three seconds to approve.
Rejection remains available with stale tracking. This freshness gate does not
prove printed-tag calibration accuracy. A proposal immediately stops speech and
takes gesture priority. Paths are kinematic previews, not measured clearance or
balance guarantees. Local mock animation requires explicit `demoMode=true` and
no dashboard review token; keep it off during robot operation.

## Speech

Without a configured live voice relay, double right pinch starts Spectacles ASR;
double right pinch again submits the transcript to the dashboard agent. Double
left pinch cancels dictation. ASR requires microphone/speech permission and
internet access. It can start before tag registration, because speaking cannot
approve a motion. Transient network errors allow retry; a denied permission
requires changing permissions or relaunching the Lens.

For GPT-Live conversation using the glasses microphone and opt-in R1 speaker,
follow [VOICE.md](VOICE.md). The same paired dashboard connection carries PCM,
voice events and trajectories. Its live-voice capability selects the audio path;
a dashboard without `--voice-url` continues using ASR even if `useLiveVoice` is
set. A separate plan-feed process is unnecessary for the integrated workflow.
Voice can request planning, but always leaves approval to the operator.

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


## Standalone development feeds

`plan_feed.py` remains a read-only plan/telemetry viewer. It accepts resolved
schema-version-1 joint plans and optional measured base odometry, and can show
remaining paths from measured joints. It does not execute robot motion.

```sh
.venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json --print
.venv/bin/python spectacles/plan_feed.py sim/plans/arm_lift_dryrun.json --host 127.0.0.1
```

For an explicit mock, run `python spectacles/trajectory_server.py --static`.
The dashboard bridge, standalone feed and mock server all default to port 8765;
run only one listener there. The legacy file-review and voice-inbox adapters
require `--review-token-file` or `--pairing-file`; they are retained for offline
experiments, not an alternative live execution path. The retired Tk UI does not
consume the voice inbox. Standalone paired live audio remains available as
described in [VOICE.md](VOICE.md).

## Verification limits

Python and Node tests exercise real protocol functions with fake sockets, scene
objects and audio providers. They cover exact review identity, tracking age,
revocation, reconnect, microphone readiness and cleanup, ASR retries and review
priority. No hardware/model request is part of those tests. Lens Studio compile,
on-device permissions, printed-tag alignment, Wi-Fi conditions and actual R1
speaker output still require commissioning on the equipment.
