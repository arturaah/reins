// R1 hand paths in robot_base or initial-pose map metres, anchored by shoulder image markers.
// Robot: x forward, y left, z up. Shoulder tags lie flat, facing upward.
// Seeing both shoulders establishes yaw regardless of each paper's rotation.
// @input Asset.InternetModule internetModule
// @input Asset.Material leftMaterial
// @input Asset.Material rightMaterial
// @input Asset.AudioTrackAsset tagDetectedSound
// @input SceneObject cameraObject
// @input Component.MarkerTrackingComponent leftMarker
// @input Component.MarkerTrackingComponent rightMarker
// @input string websocketUrl = "ws://127.0.0.1:8765"
// @input string fallbackWebsocketUrl = ""
// @input float tagForwardM = 0.03
// @input float tagSideM = 0.13
// @input float tagHeightM = 1.019
// @input float pathRadiusCm = 0.65
// @input bool allowTemporaryAnchor = false

var root = script.getSceneObject();
var anchor = global.scene.createSceneObject("R1 robot base (world anchored)");
anchor.setParent(root);
var leftVisual = makeVisual("Left hand trajectory", script.leftMaterial);
var rightVisual = makeVisual("Right hand trajectory", script.rightMaterial);
var socket = null;
var socketUrlIndex = 0;
var reconnectAt = 0;
var lastReceivedAt = -1000;
var lastMockAt = -1000;
var hasReceivedTrajectory = false;
var latestTrajectory = null;
var hasAnchor = false;
var tagAnchored = false;
var lastTagName = "";
var lastNetworkError = "";
var liveLogged = false;
var markerVisibleLastFrame = false;
var observedTags = {left:null, right:null};
// Lens Studio's image-marker pose scale follows its configured printed size.
// A pair of shoulder centres lets us correct for cut-out tags printed smaller
// or larger than the 10 cm asset setting without changing the robot geometry.
var worldCmPerRobotCm = 1;
var statusObject = null;
var statusVisual = null;
var statusFrame = null;
var statusTextObject = null;
var statusText = null;
var statusUntil = 0;
var statusPosition = null;
var notificationAudio = null;
var lastNotificationAt = -1000;
var pendingReview = null;
var reviewChoice = "";
var reviewChoiceAt = -1000;
var reviewSent = false;
var reviewMessage = "";
var reviewObject = null;
var reviewText = null;
var reviewFrame = null;
var socketReady = false;
var asrModule = null;
var voiceListening = false;
var voiceFinal = [];
var voicePartial = "";
var voiceSendRequestedAt = -1;
var voiceStartedAt = -1;
var voiceMessage = "";
var voiceMessageUntil = 0;
var voicePinchAt = -1000;
var voiceCommandId = "";
var voiceSequence = 0;
var voiceGestureLockUntil = 0;
var microphoneBlocked = false;
var voiceSessionId = 0;

function voiceText() {
    return (voiceFinal.join(" ") + " " + voicePartial).trim();
}
function voiceNotice(message) {
    voiceMessage = message;
    voiceMessageUntil = getTime() + 5;
    print("R1 AR voice: " + message);
}
function stopVoice(send) {
    if (!voiceListening) { return; }
    if (send) {
        // Keep the recognizer alive long enough to receive its final update.
        // An interim update may be the only usable result on a slow network.
        if (voiceSendRequestedAt < 0) { voiceSendRequestedAt = getTime(); }
        voiceNotice("FINISHING SPEECH");
        return;
    }
    voiceListening = false;
    voiceSessionId++;
    voiceSendRequestedAt = -1;
    try { asrModule.stopTranscribing(); } catch (e) { print("R1 AR voice stop: " + e); }
    voiceNotice("VOICE CANCELLED");
}
function startVoice() {
    if (!tagAnchored) { voiceNotice("SCAN BOTH SHOULDER TAGS FIRST"); return; }
    if (!socketReady) { voiceNotice("CONNECT TO ARTUR FIRST"); return; }
    if (microphoneBlocked) { voiceNotice("MICROPHONE PERMISSION DENIED"); return; }
    if (!asrModule) { voiceNotice("MICROPHONE UNAVAILABLE"); return; }
    voiceFinal = []; voicePartial = ""; voiceSendRequestedAt = -1;
    var sessionId = ++voiceSessionId;
    try {
        var options = AsrModule.AsrTranscriptionOptions.create();
        options.mode = AsrModule.AsrMode.HighAccuracy;
        options.silenceUntilTerminationMs = 1200;
        options.onTranscriptionUpdateEvent.add(function(update) {
            if (!voiceListening || sessionId !== voiceSessionId) { return; }
            print("R1 AR voice: ASR update final=" + !!update.isFinal +
                  " chars=" + String(update.text || "").length);
            if (update.isFinal) {
                if (update.text) {
                    voiceFinal.push(update.text);
                    voicePartial = "";
                }
            } else if (update.text) {
                // Empty interim updates are common between recognition passes.
                // They must not erase the only phrase we have heard.
                voicePartial = update.text;
            }
            if (voiceSendRequestedAt >= 0 && update.isFinal && voiceText()) {
                finishVoiceSend();
            }
        });
        options.onTranscriptionErrorEvent.add(function(code) {
            if (sessionId !== voiceSessionId) { return; }
            voiceListening = false;
            voiceSessionId++;
            microphoneBlocked = true;
            voiceNotice("SPEECH ERROR " + code);
        });
        voiceListening = true;
        voiceStartedAt = getTime();
        asrModule.startTranscribing(options);
        voiceNotice("LISTENING");
    } catch (e) {
        voiceListening = false;
        voiceSessionId++;
        microphoneBlocked = true;
        voiceNotice("MICROPHONE PERMISSION DENIED");
        print("R1 AR voice start failed: " + e);
    }
}
function finishVoiceSend() {
    if (!voiceListening) { return; }
    voiceListening = false;
    voiceSessionId++;
    voiceSendRequestedAt = -1;
    var phrase = voiceText().slice(0, 500);
    try { asrModule.stopTranscribing(); } catch (e) { print("R1 AR voice stop: " + e); }
    if (!phrase) { voiceNotice("NO SPEECH RESULT - CHECK MIC/INTERNET"); return; }
    if (!socketReady) { voiceNotice("NO NETWORK"); return; }
    voiceCommandId = "spectacles-" + Date.now() + "-" + (++voiceSequence);
    try {
        socket.send(JSON.stringify({type:"voice_command", version:1,
                                    id:voiceCommandId, text:phrase}));
        voiceNotice("SENDING: " + phrase.slice(0, 44));
    } catch (e) { voiceNotice("VOICE SEND FAILED"); }
}
function voicePinch(side) {
    if (pendingReview) {
        if (voiceListening) { stopVoice(false); }
        reviewPinch(side === "right" ? "approve" : "decline");
        return;
    }
    if (!tagAnchored) {
        if (side === "right") { voiceNotice("SCAN BOTH SHOULDER TAGS FIRST"); }
        return;
    }
    if (side === "left") {
        if (voiceListening) { stopVoice(false); }
        return;
    }
    var now = getTime();
    if (now < voiceGestureLockUntil) { return; }
    if (now - voicePinchAt > 0.25 && now - voicePinchAt < 4) {
        voicePinchAt = -1000;
        voiceGestureLockUntil = now + 1.5;
        if (voiceListening) { stopVoice(true); } else { startVoice(); }
    } else {
        voicePinchAt = now;
        voiceNotice(voiceListening ? "RIGHT PINCH AGAIN TO SEND" : "RIGHT PINCH AGAIN TO SPEAK");
    }
}

try {
    if (script.tagDetectedSound) {
        notificationAudio = root.createComponent("Component.AudioComponent");
        notificationAudio.audioTrack = script.tagDetectedSound;
        notificationAudio.volume = 1;
        print("R1 AR: tag notification sound ready");
    } else {
        print("R1 AR: tag notification sound missing");
    }
} catch (e) { print("R1 AR: notification sound unavailable: " + e); }

// Draw the cue with the same mesh/material path as the visible trajectories.
try {
    statusObject = global.scene.createSceneObject("R1 tracking status HUD");
    statusObject.setParent(root);
    statusVisual = statusObject.createComponent("Component.RenderMeshVisual");
    statusVisual.mainMaterial = script.leftMaterial;
    statusVisual.enabled = false;
    statusFrame = statusObject.createComponent("Component.RenderMeshVisual");
    statusFrame.mainMaterial = script.leftMaterial;
    statusFrame.enabled = false;
    var canvasObject = global.scene.createSceneObject("R1 notification canvas");
    canvasObject.setParent(statusObject);
    var canvas = canvasObject.createComponent("Component.Canvas");
    canvas.setSize(new vec2(48, 18));
    statusTextObject = global.scene.createSceneObject("R1 notification text");
    statusTextObject.setParent(canvasObject);
    var textScreen = statusTextObject.createComponent("Component.ScreenTransform");
    textScreen.anchors.setSize(new vec2(1.8, 1.5));
    statusText = statusTextObject.createComponent("Component.Text");
    statusText.text = "LEFT SHOULDER";
    statusText.size = 175;
    statusText.horizontalAlignment = HorizontalAlignment.Center;
    statusText.verticalAlignment = VerticalAlignment.Center;
    statusText.textFill.color = new vec4(0.2, 1, 0.9, 1);
    statusTextObject.enabled = false;
} catch (e) {
    print("R1 AR: tracking HUD unavailable: " + e);
}

// Head-following review card. Gestures come from Spectacles' built-in GestureModule;
// the server still checks the proposal ID and the exact plan hash.
try {
    reviewObject = global.scene.createSceneObject("R1 proposal review");
    reviewObject.setParent(root);
    reviewFrame = reviewObject.createComponent("Component.RenderMeshVisual");
    reviewFrame.mainMaterial = script.leftMaterial;
    var reviewCanvasObject = global.scene.createSceneObject("R1 review canvas");
    reviewCanvasObject.setParent(reviewObject);
    var reviewCanvas = reviewCanvasObject.createComponent("Component.Canvas");
    reviewCanvas.setSize(new vec2(80, 32));
    var reviewTextObject = global.scene.createSceneObject("R1 review text");
    reviewTextObject.setParent(reviewCanvasObject);
    var reviewScreen = reviewTextObject.createComponent("Component.ScreenTransform");
    reviewScreen.anchors.setSize(new vec2(2.0, 2.0));
    reviewText = reviewTextObject.createComponent("Component.Text");
    reviewText.size = 130;
    reviewText.horizontalAlignment = HorizontalAlignment.Center;
    reviewText.verticalAlignment = VerticalAlignment.Center;
    reviewText.textFill.color = new vec4(0.2, 1, 0.9, 1);
    drawTube([[-42,-18,0],[42,-18,0],[42,18,0],[-42,18,0],[-42,-18,0]],
        reviewFrame, function(p) { return p; }, 1.2);
    reviewObject.enabled = false;
} catch (e) { print("R1 AR: review card unavailable: " + e); }

function reviewPinch(choice) {
    if (!pendingReview || !tagAnchored || !socketReady || reviewSent) { return; }
    var now = getTime();
    if (reviewChoice === choice && now - reviewChoiceAt > 0.25 && now - reviewChoiceAt < 4) {
        try {
            socket.send(JSON.stringify({type:"review_decision", version:1,
                                        id:pendingReview.id, decision:choice}));
            reviewSent = true;
            reviewMessage = "SENDING " + choice.toUpperCase();
            print("R1 AR: " + choice + " sent for proposal " + pendingReview.id);
        } catch (e) { reviewMessage = "SEND FAILED"; print("R1 AR: review send failed: " + e); }
        return;
    }
    reviewChoice = choice;
    reviewChoiceAt = now;
    reviewMessage = "PINCH " + (choice === "approve" ? "RIGHT" : "LEFT") + " AGAIN TO " +
                    (choice === "approve" ? "ACCEPT" : "REJECT");
}

try {
    var gestureModule = require('LensStudio:GestureModule');
    gestureModule.getPinchDownEvent(GestureModule.HandType.Right).add(function() { voicePinch("right"); });
    gestureModule.getPinchDownEvent(GestureModule.HandType.Left).add(function() { voicePinch("left"); });
    print("R1 AR: right double-pinch speaks or accepts; left cancels or rejects");
} catch (e) { print("R1 AR: review gesture unavailable: " + e); }
try {
    asrModule = require('LensStudio:AsrModule');
    print("R1 AR: Spectacles speech recognition ready");
} catch (e) { print("R1 AR: speech recognition unavailable: " + e); }

function showStatus(found, side) {
    if (!statusVisual) { return; }
    statusVisual.mainMaterial = found ? script.leftMaterial : script.rightMaterial;
    statusVisual.enabled = true;
    statusFrame.enabled = true;
    statusFrame.mainMaterial = found ? script.leftMaterial : script.rightMaterial;
    if (statusTextObject) {
        statusText.text = found ? side.toUpperCase() + " SHOULDER" : "TAG LOST";
        statusTextObject.enabled = true;
    }
    // Shape coordinates are metres in the camera-facing plane.
    drawTube(found ? [[-0.12,0,0],[-0.04,-0.08,0],[0.14,0.11,0]] :
                     [[-0.11,-0.11,0],[0.11,0.11,0],[-0.11,0.11,0],
                      [0.11,-0.11,0]], statusVisual, function(p) {
        return [p[0]*100,p[1]*100,p[2]*100];
    }, 1.5);
    drawTube([[-23,-10,0],[23,-10,0],[23,10,0],[-23,10,0],[-23,-10,0]],
        statusFrame, function(p) { return p; }, 0.8);
    statusUntil = getTime() + 4;
}

function makeVisual(name, material) {
    var object = global.scene.createSceneObject(name);
    object.setParent(anchor);
    var visual = object.createComponent("Component.RenderMeshVisual");
    if (material) { visual.mainMaterial = material; }
    return visual;
}

// Rendering basis: forward -> +y, robot-left -> -x, up -> +z.
// The basis is built from the two tag centres, not their printed orientations.
function robotToMarker(p) {
    var cm = 100 * worldCmPerRobotCm;
    return [-p[1] * cm, p[0] * cm, p[2] * cm];
}

function observeMarker(tracker, name) {
    if (!tracker || !tracker.isTracking()) { return null; }
    var t = tracker.getTransform();
    var rotation = t.getWorldRotation();
    var pos = t.getWorldPosition();
    observedTags[name] = {position:pos, time:getTime()};
    // Keep the brief label just above the recognized tag as the wearer moves.
    statusPosition = pos.add(
        rotation.multiplyVec3(new vec3(0, 0, 18)));
    var newlySeen = !markerVisibleLastFrame || lastTagName !== name;
    if (newlySeen) {
        print("R1 AR: APRILTAG DETECTED: " + name + " shoulder");
        showStatus(true, name);
    }
    if (newlySeen) {
        if (getTime() - lastNotificationAt > 2) {
            if (notificationAudio) {
                notificationAudio.play(1);
                print("R1 AR: tag notification sound played");
            }
            lastNotificationAt = getTime();
        }
    }
    lastTagName = name;
    markerVisibleLastFrame = true;
    return pos;
}

function placeAnchor(position, rotation) {
    var base = anchor.getTransform();
    var firstCalibration = !tagAnchored;
    if (!tagAnchored) {
        base.setWorldPosition(position);
        base.setWorldRotation(rotation);
    } else {
        base.setWorldPosition(vec3.lerp(base.getWorldPosition(), position, 0.15));
        base.setWorldRotation(quat.slerp(base.getWorldRotation(), rotation, 0.15));
    }
    tagAnchored = true;
    hasAnchor = true;
    if (firstCalibration && latestTrajectory) { applyTrajectory(latestTrajectory); }
}

function calibrateFromBothTags() {
    var left = observedTags.left, right = observedTags.right;
    if (!left || !right || Math.abs(left.time-right.time) > 10) { return false; }
    var a=left.position, b=right.position;
    // Spectacles world has +y vertical. R1 robot-left is right-tag -> left-tag.
    var lateral = [a.x-b.x, 0, a.z-b.z];
    var span = Math.sqrt(lateral[0]*lateral[0]+lateral[2]*lateral[2]);
    if (span < 8 || span > 60) { return false; } // cm; reject unrelated detections
    var expectedSpan = 2 * script.tagSideM * 100;
    if (expectedSpan <= 0) { return false; }
    var measuredScale = span / expectedSpan;
    if (measuredScale < 0.35 || measuredScale > 2.5) { return false; }
    if (Math.abs(measuredScale - worldCmPerRobotCm) > 0.01) {
        worldCmPerRobotCm = measuredScale;
        if (latestTrajectory) { applyTrajectory(latestTrajectory); }
        print("R1 AR: shoulder span " + span.toFixed(1) +
              " world cm; render scale " + measuredScale.toFixed(2));
    }
    var leftAxis = unit(lateral), up = [0,1,0];
    var forward = unit(cross(leftAxis, up));
    var rotation = quat.fromRotationMat4(mat4.makeBasis(
        new vec3(-leftAxis[0],-leftAxis[1],-leftAxis[2]),
        new vec3(forward[0],forward[1],forward[2]), new vec3(0,1,0)));
    var centre = new vec3((a.x+b.x)/2,(a.y+b.y)/2,(a.z+b.z)/2);
    var offset=robotToMarker([script.tagForwardM,0,script.tagHeightM]);
    var origin=centre.sub(rotation.multiplyVec3(new vec3(offset[0],offset[1],offset[2])));
    var firstPair = !tagAnchored;
    placeAnchor(origin,rotation);
    if (firstPair) { print("R1 AR: BOTH SHOULDERS CALIBRATED; robot forward from tag baseline"); }
    return true;
}

function updateAnchor() {
    var left=observeMarker(script.leftMarker,"left");
    var right=observeMarker(script.rightMarker,"right");
    if (left || right) {
        // A map path is fixed to the room at the initial robot pose. Moving
        // shoulder tags must not drag the future path along with the robot.
        if (tagAnchored && latestTrajectory && latestTrajectory.frame === "map") { return; }
        if (!calibrateFromBothTags() && tagAnchored) {
            // A single visible shoulder can update translation after a pair
            // established orientation. Rescan both after the robot turns.
            var name=left ? "left" : "right", pos=left || right;
            var side=name==="left" ? 1 : -1;
            var base=anchor.getTransform(), rotation=base.getWorldRotation();
            var offset=robotToMarker([script.tagForwardM,side*script.tagSideM,
                                      script.tagHeightM]);
            placeAnchor(pos.sub(rotation.multiplyVec3(
                new vec3(offset[0],offset[1],offset[2]))),rotation);
        }
        return;
    }
    if (markerVisibleLastFrame) {
        print("R1 AR: shoulder tag lost; holding last calibrated world pose");
        // Keep the success box visible for its full duration. A momentary
        // tracking loss must not overwrite it before the wearer can see it.
        if (!tagAnchored || getTime() > statusUntil) { showStatus(false); }
        markerVisibleLastFrame = false;
    }
    if (hasAnchor || !script.allowTemporaryAnchor || !script.cameraObject) { return; }
    var camera = script.cameraObject.getTransform();
    anchor.getTransform().setWorldRotation(
        camera.getWorldRotation().multiply(quat.angleAxis(Math.PI, vec3.up())));
    anchor.getTransform().setWorldPosition(
        camera.getWorldPosition().add(camera.back.uniformScale(150)));
    hasAnchor = true;
    print("R1 AR: temporary anchor set; view both shoulder tags to calibrate");
}

function isPoint(p) {
    return Array.isArray(p) && p.length === 3 && p.every(function (n) {
        return typeof n === "number" && isFinite(n) && Math.abs(n) < 10;
    });
}
function isPath(points) {
    return Array.isArray(points) && points.length >= 2 && points.length <= 512 &&
           points.every(isPoint);
}
function isTrajectoryPath(points) {
    return Array.isArray(points) && points.length <= 512 && points.every(isPoint);
}
function cross(a, b) {
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function unit(v) {
    var len = Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    return len > 0.0001 ? [v[0]/len, v[1]/len, v[2]/len] : null;
}
function add(a, b, scale) {
    return [a[0]+b[0]*scale, a[1]+b[1]*scale, a[2]+b[2]*scale];
}

function sub(a, b) {
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
}
function frame(dir) {
    var side = unit(cross(dir, [0,1,0])) || unit(cross(dir, [1,0,0]));
    return {side: side, up: unit(cross(dir, side))};
}
function pathLength(cm) {
    var total = 0;
    for (var i=1; i<cm.length; i++) {
        var d = sub(cm[i], cm[i-1]);
        total += Math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    }
    return total;
}
// The path up to the point `len` cm of arc before its end, so a tube stops where an
// arrowhead's base sits. null when the path is shorter than len.
function trimEnd(cm, len) {
    var remaining = len;
    for (var i=cm.length-1; i>0; i--) {
        var a = cm[i-1], b = cm[i], d = sub(b, a);
        var seg = Math.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
        if (seg >= remaining) {
            var t = seg > 0 ? (seg - remaining) / seg : 0;
            var kept = cm.slice(0, i);
            kept.push(add(a, d, t));
            return kept;
        }
        remaining -= seg;
    }
    return null;
}

// A four-sided tube is visible from any angle and does not depend on line width.
// withArrow ends the tube in a pyramid whose apex is the last point, so the direction of travel is visible.
function drawTube(points, visual, pointToCm, radiusOverride, withArrow) {
    if (!isPath(points)) { return; }
    var vertices = [], indices = [];
    var radius = radiusOverride || Math.max(0.15, Math.min(3, script.pathRadiusCm));
    var convert = pointToCm || robotToMarker;
    var cm = points.map(function (p) { return convert(p); });
    var tube = cm, head = null;
    if (withArrow) {
        var total = pathLength(cm);
        var headLen = Math.min(Math.max(3, radius * 5), total / 2);
        var trimmed = total >= 1 ? trimEnd(cm, headLen) : null;
        var headDir = trimmed ? unit(sub(cm[cm.length-1], trimmed[trimmed.length-1])) : null;
        if (headDir) {
            tube = trimmed;
            head = {tip: cm[cm.length-1], dir: headDir, len: headLen,
                    radius: Math.min(radius * 2.5, headLen * 0.5)};
        }
    }
    for (var i=0; i<tube.length-1; i++) {
        var a = tube[i], b = tube[i+1];
        var dir = unit(sub(b, a));
        if (!dir) { continue; }
        var f = frame(dir);
        var corners = [f.side, f.up, [-f.side[0],-f.side[1],-f.side[2]],
                       [-f.up[0],-f.up[1],-f.up[2]]];
        var base = vertices.length / 3;
        for (var ring=0; ring<2; ring++) {
            var centre = ring === 0 ? a : b;
            for (var c=0; c<4; c++) {
                var p = add(centre, corners[c], radius);
                vertices.push(p[0],p[1],p[2]);
            }
        }
        for (var edge=0; edge<4; edge++) {
            var next=(edge+1)%4;
            indices.push(base+edge,base+next,base+4+edge,
                         base+next,base+4+next,base+4+edge);
        }
    }
    if (head) {
        var hf = frame(head.dir);
        var headBase = add(head.tip, head.dir, -head.len);
        var headCorners = [hf.side, hf.up, [-hf.side[0],-hf.side[1],-hf.side[2]],
                           [-hf.up[0],-hf.up[1],-hf.up[2]]];
        var h0 = vertices.length / 3;
        for (var hc=0; hc<4; hc++) {
            var hp = add(headBase, headCorners[hc], head.radius);
            vertices.push(hp[0],hp[1],hp[2]);
        }
        vertices.push(head.tip[0],head.tip[1],head.tip[2]);
        for (var he=0; he<4; he++) { indices.push(h0+he, h0+(he+1)%4, h0+4); }
        indices.push(h0, h0+2, h0+1, h0, h0+3, h0+2);   // base cap, facing back along the path
    }
    if (!indices.length) { return; }
    var mesh = new MeshBuilder([{name:"position",components:3}]);
    mesh.topology = MeshTopology.Triangles;
    mesh.indexType = MeshIndexType.UInt16;
    mesh.appendVerticesInterleaved(vertices);
    mesh.appendIndices(indices);
    if (!mesh.isValid()) { print("R1 AR: invalid mesh"); return; }
    visual.mesh = mesh.getMesh();
    mesh.updateMesh();
}

function drawPoint(point, visual) {
    var c=robotToMarker(point);
    var r=Math.max(0.5,Math.min(4,script.pathRadiusCm*2.2));
    var vertices=[c[0]+r,c[1],c[2], c[0]-r,c[1],c[2],
                  c[0],c[1]+r,c[2], c[0],c[1]-r,c[2],
                  c[0],c[1],c[2]+r, c[0],c[1],c[2]-r];
    var indices=[2,4,0, 2,1,4, 2,5,1, 2,0,5,
                 3,0,4, 3,4,1, 3,1,5, 3,5,0];
    var mesh=new MeshBuilder([{name:"position",components:3}]);
    mesh.topology=MeshTopology.Triangles;
    mesh.indexType=MeshIndexType.UInt16;
    mesh.appendVerticesInterleaved(vertices);
    mesh.appendIndices(indices);
    if (!mesh.isValid()) { print("R1 AR: invalid hand point mesh"); return; }
    visual.mesh=mesh.getMesh();
    mesh.updateMesh();
}

function applyTrajectory(message) {
    if (!message || message.type !== "trajectory" || message.version !== 1 ||
        (message.frame !== "robot_base" && message.frame !== "map") ||
        message.units !== "m" ||
        !message.hands || !isTrajectoryPath(message.hands.left) ||
        !isTrajectoryPath(message.hands.right)) {
        print("R1 AR: rejected incompatible trajectory"); return;
    }
    if (latestTrajectory && latestTrajectory.frame !== message.frame) {
        // A new frame needs a new scan. In particular, a map plan must be
        // anchored at the robot's starting location before it walks.
        tagAnchored = false;
        observedTags = {left:null, right:null};
        print("R1 AR: coordinate frame changed; rescan both shoulder tags");
    }
    var incoming = message.review;
    if (incoming && typeof incoming.id === "string" && incoming.id.length >= 16 &&
        (incoming.mode === "live" || incoming.mode === "dry-run" || incoming.mode === "sim")) {
        if (!pendingReview || pendingReview.id !== incoming.id) {
            // A new proposal takes the gesture and HUD priority immediately.
            if (voiceListening) { stopVoice(false); }
            voicePinchAt = -1000;
            voiceGestureLockUntil = 0;
            reviewChoice = ""; reviewSent = false; reviewMessage = "";
            print("R1 AR: proposal ready: " + incoming.text);
        }
        pendingReview = incoming;
    } else {
        if (pendingReview) { voicePinchAt = -1000; }
        pendingReview = null; reviewChoice = ""; reviewSent = false; reviewMessage = "";
    }
    latestTrajectory = message;
    function drawHand(points, visual) {
        visual.enabled = points.length > 0;
        if (!points.length) { return; }
        if (points.length === 1 || points.every(function(p) {
            var a=points[0];
            return Math.abs(p[0]-a[0])+Math.abs(p[1]-a[1])+Math.abs(p[2]-a[2]) < 0.001;
        })) {
            drawPoint(points[0], visual);
            return;
        }
        drawTube(points, visual, null, 0, true);   // arrowhead at the destination
    }
    drawHand(message.hands.left, leftVisual);
    drawHand(message.hands.right, rightVisual);
}
function localMock(phase) {
    var hands={left:[],right:[]};
    for (var i=0; i<30; i++) {
        // Start at the neutral hand sites in R1_fixed_base.xml, not at a tag.
        var t=i/29, x=0.2909+0.25*t;
        var z=0.771+0.18*t+0.03*t*Math.sin(Math.PI*t+phase);
        hands.left.push([x,0.1386+0.06*t,z]);
        hands.right.push([x,-0.1386-0.06*t,z]);
    }
    return {type:"trajectory",version:1,id:"local-mock",frame:"robot_base",
            units:"m",hands:hands};
}
function connect() {
    if (!script.internetModule) { return; }
    var urls = [script.websocketUrl];
    if (script.fallbackWebsocketUrl &&
        script.fallbackWebsocketUrl !== script.websocketUrl) {
        urls.push(script.fallbackWebsocketUrl);
    }
    var url = urls[socketUrlIndex % urls.length];
    try {
        socket=script.internetModule.createWebSocket(url);
        socket.onopen=function(){socketReady=true;lastNetworkError="";print("R1 AR: WebSocket connected to "+url);};
        socket.onmessage=function(event){
            try {
                var trajectory = JSON.parse(event.data);
                if (trajectory.type === "review_ack") {
                    if (pendingReview && trajectory.id === pendingReview.id) {
                        reviewMessage = trajectory.accepted ? "DECISION RECEIVED" : "PROPOSAL EXPIRED";
                        if (!trajectory.accepted) { reviewSent = false; reviewChoice = ""; }
                    }
                    return;
                }
                if (trajectory.type === "voice_ack") {
                    if (trajectory.id === voiceCommandId) {
                        voiceNotice(trajectory.accepted ? "COMMAND QUEUED FOR CLAUDE" : "ARTUR BUSY; TRY AGAIN");
                    }
                    return;
                }
                applyTrajectory(trajectory);
                lastReceivedAt=getTime();
                if (trajectory && trajectory.type === "trajectory" &&
                    (trajectory.frame === "robot_base" || trajectory.frame === "map") &&
                    trajectory.hands &&
                    isTrajectoryPath(trajectory.hands.left) &&
                    isTrajectoryPath(trajectory.hands.right)) {
                    hasReceivedTrajectory = true;
                }
                if (!liveLogged && trajectory && trajectory.hands &&
                    isTrajectoryPath(trajectory.hands.left) &&
                    isTrajectoryPath(trajectory.hands.right)) {
                    print("R1 AR: live trajectory received ("+
                          trajectory.hands.left.length+" points per hand)");
                    liveLogged=true;
                }
            } catch(e) { print("R1 AR: invalid message: "+e); }
        };
        socket.onerror=function(event){
            if (lastNetworkError!=="connection") {
                print("R1 AR: WebSocket connection failed at "+url+
                      "; trying next route ("+event+")");
                lastNetworkError="connection";
            }
        };
        socket.onclose=function(event){
            socketReady=false;socket=null;socketUrlIndex++;reconnectAt=getTime()+2;
            if (voiceListening) { stopVoice(false); }
            if (event && event.code && event.code!==1000 && !lastNetworkError) {
                print("R1 AR: WebSocket closed with code "+event.code);
                lastNetworkError="closed";
            }
        };
    } catch(e) {
        socketReady=false;
        if (lastNetworkError!=="unavailable") {
            print("R1 AR: WebSocket unavailable: "+e);
            lastNetworkError="unavailable";
        }
        socket=null;
        socketUrlIndex++;
        reconnectAt=String(e).indexOf("simulated platform")>=0 ? Infinity : getTime()+2;
    }
}
script.createEvent("UpdateEvent").bind(function(){
    updateAnchor();
    if (voiceListening && voiceSendRequestedAt >= 0) {
        var voiceWait = getTime() - voiceSendRequestedAt;
        if (voiceWait > 1.5 && voiceText()) {
            finishVoiceSend();
        } else if (voiceWait > 6) {
            stopVoice(false);
            voiceNotice("NO SPEECH RESULT - CHECK MIC/INTERNET");
        }
    }
    if (statusObject && script.cameraObject && statusPosition) {
        var head = script.cameraObject.getTransform();
        var label = statusObject.getTransform();
        label.setWorldPosition(statusPosition);
        // Billboard the text towards the wearer while its position stays on the tag.
        label.setWorldRotation(head.getWorldRotation());
    }
    if (statusVisual && getTime() > statusUntil) {
        statusVisual.enabled = false;
        if (statusFrame) { statusFrame.enabled = false; }
        if (statusTextObject) { statusTextObject.enabled = false; }
    }
    if (reviewObject && script.cameraObject) {
        reviewObject.enabled = !!pendingReview || voiceListening || getTime() < voiceMessageUntil;
        if (reviewObject.enabled) {
            var cameraTransform = script.cameraObject.getTransform();
            var panel = reviewObject.getTransform();
            panel.setWorldPosition(cameraTransform.getWorldPosition().add(
                cameraTransform.back.uniformScale(140)));
            panel.setWorldRotation(cameraTransform.getWorldRotation());
            if (reviewChoice && !reviewSent && getTime() - reviewChoiceAt >= 4) {
                reviewChoice = ""; reviewMessage = "";
            }
            reviewText.text = pendingReview ?
                "REVIEW " + pendingReview.mode.toUpperCase() + "\n" +
                String(pendingReview.text || "NEW PATH").slice(0, 38) + "\n" +
                (reviewMessage || (!tagAnchored ? "SCAN BOTH TAGS FIRST" :
                 !socketReady ? "WAITING FOR CONNECTION" :
                 "RIGHT x2 ACCEPT   LEFT x2 REJECT")) :
                (voiceListening ? (voiceSendRequestedAt >= 0 ? "WAITING FOR WORDS\n" : "LISTENING\n") +
                 (voiceText() || (getTime() - voiceStartedAt > 3 ? "NO WORDS YET" : "SPEAK NOW")).slice(-55) +
                 "\nRIGHT x2 SEND   LEFT CANCEL" : voiceMessage);
        }
    }
    if (!socket && getTime()>=reconnectAt) { connect(); }
    // Keep the last real plan on screen when USB/Wi-Fi drops. Only animate a
    // mock before any robot-frame trajectory has arrived this Lens session.
    if (!hasReceivedTrajectory && getTime()-lastReceivedAt>1.5 &&
        getTime()-lastMockAt>0.3) {
        applyTrajectory(localMock(getTime()*0.3));
        lastMockAt=getTime();
    }
    if (!tagAnchored) {
        leftVisual.enabled = false;
        rightVisual.enabled = false;
    }
});
