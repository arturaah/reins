// R1 hand paths in robot_base metres, anchored by shoulder image markers.
// Robot: x forward, y left, z up. Marker: x image-right, y up, +z outward.
// @input Asset.InternetModule internetModule
// @input Asset.Material leftMaterial
// @input Asset.Material rightMaterial
// @input Asset.AudioTrackAsset tagDetectedSound
// @input SceneObject cameraObject
// @input Component.MarkerTrackingComponent leftMarker
// @input Component.MarkerTrackingComponent rightMarker
// @input string websocketUrl = "ws://127.0.0.1:8765"
// @input float tagForwardM = 0.04
// @input float tagSideM = 0.16
// @input float tagHeightM = 1.02
// @input bool previewFromTags = true
// @input float previewTagHalfSpacingM = 0.071
// @input float pathRadiusCm = 0.65
// @input bool allowTemporaryAnchor = true

var root = script.getSceneObject();
var anchor = global.scene.createSceneObject("R1 robot base (world anchored)");
anchor.setParent(root);
var leftVisual = makeVisual("Left hand trajectory", script.leftMaterial);
var rightVisual = makeVisual("Right hand trajectory", script.rightMaterial);
var socket = null;
var reconnectAt = 0;
var lastReceivedAt = -1000;
var lastMockAt = -1000;
var hasAnchor = false;
var tagAnchored = false;
var lastTagName = "";
var lastNetworkError = "";
var liveLogged = false;
var markerVisibleLastFrame = false;
var statusObject = null;
var statusVisual = null;
var statusFrame = null;
var statusTextObject = null;
var statusText = null;
var statusUntil = 0;
var statusPosition = null;
var notificationAudio = null;
var lastNotificationAt = -1000;
var liveRobotCoordinates = false;

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

// Upright tag on the robot's front: forward -> marker +z, left -> +x, up -> +y.
function robotToMarker(p) { return [p[1] * 100, p[2] * 100, p[0] * 100]; }
function previewFromTags() { return script.previewFromTags !== false; }
function markerSideM() {
    return previewFromTags() && !liveRobotCoordinates ?
        (script.previewTagHalfSpacingM || 0.071) : script.tagSideM;
}

function useMarker(tracker, side, name) {
    if (!tracker || !tracker.isTracking()) { return false; }
    var t = tracker.getTransform();
    var rotation = t.getWorldRotation();
    // Keep the brief label just above the recognized tag as the wearer moves.
    statusPosition = t.getWorldPosition().add(
        rotation.multiplyVec3(new vec3(0, 18, 3)));
    var localOffset = robotToMarker([script.tagForwardM,
                                     side * markerSideM(), script.tagHeightM]);
    var offsetWorld = rotation.multiplyVec3(
        new vec3(localOffset[0], localOffset[1], localOffset[2]));
    var basePosition = t.getWorldPosition().sub(offsetWorld);
    var base = anchor.getTransform();
    var newlySeen = !markerVisibleLastFrame || lastTagName !== name;
    if (!tagAnchored || newlySeen) {
        base.setWorldPosition(basePosition);
        base.setWorldRotation(rotation);
        // Always update the side label on a switch, even during sound cooldown.
        if (newlySeen) {
            print("R1 AR: APRILTAG DETECTED: " + name + " shoulder; trajectories calibrated");
            showStatus(true, name);
        }
        if (getTime() - lastNotificationAt > 2) {
            if (notificationAudio) {
                notificationAudio.play(1);
                print("R1 AR: tag notification sound played");
            }
            lastNotificationAt = getTime();
        }
    } else {
        base.setWorldPosition(vec3.lerp(base.getWorldPosition(), basePosition, 0.15));
        base.setWorldRotation(quat.slerp(base.getWorldRotation(), rotation, 0.15));
    }
    hasAnchor = true;
    tagAnchored = true;
    lastTagName = name;
    markerVisibleLastFrame = true;
    return true;
}

function updateAnchor() {
    if (useMarker(script.leftMarker, 1, "left")) { return; }
    if (useMarker(script.rightMarker, -1, "right")) { return; }
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
    print("R1 AR: temporary anchor set; view either tag to calibrate");
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
    return (Array.isArray(points) && points.length === 0) || isPath(points);
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

// A four-sided tube is visible from any angle and does not depend on line width.
function drawTube(points, visual, pointToCm, radiusOverride) {
    if (!isPath(points)) { return; }
    var vertices = [], indices = [];
    var radius = radiusOverride || Math.max(0.15, Math.min(3, script.pathRadiusCm));
    var convert = pointToCm || robotToMarker;
    for (var i=0; i<points.length-1; i++) {
        var a = convert(points[i]), b = convert(points[i+1]);
        var dir = unit([b[0]-a[0], b[1]-a[1], b[2]-a[2]]);
        if (!dir) { continue; }
        var side = unit(cross(dir, [0,1,0])) || unit(cross(dir, [1,0,0]));
        var up = unit(cross(dir, side));
        var corners = [side, up, [-side[0],-side[1],-side[2]],
                       [-up[0],-up[1],-up[2]]];
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

function applyTrajectory(message) {
    if (!message || message.type !== "trajectory" || message.version !== 1 ||
        message.frame !== "robot_base" || message.units !== "m" ||
        !message.hands || !isTrajectoryPath(message.hands.left) ||
        !isTrajectoryPath(message.hands.right)) {
        print("R1 AR: rejected incompatible trajectory"); return;
    }
    liveRobotCoordinates = message.progress_source === "measured_joints";
    // Desk preview: translate each robot-frame polyline so its first point is
    // exactly at that shoulder marker. Keep the path shape and input frame.
    function drawHand(points, visual, side) {
        // The feed sends [] when this hand has reached the end of its plan.
        // Disabling also removes any mesh left from the previous update.
        visual.enabled = points.length > 0;
        if (!points.length) { return; }
        // Live robot positions must never be shifted back to a tag centre.
        if (!previewFromTags() || liveRobotCoordinates) {
            drawTube(points, visual); return;
        }
        var start = points[0];
        drawTube(points, visual, function(p) {
            return robotToMarker([
                p[0] - start[0] + script.tagForwardM,
                p[1] - start[1] + side * markerSideM(),
                p[2] - start[2] + script.tagHeightM
            ]);
        });
    }
    drawHand(message.hands.left, leftVisual, 1);
    drawHand(message.hands.right, rightVisual, -1);
}
function localMock(phase) {
    var hands={left:[],right:[]};
    for (var i=0; i<30; i++) {
        var t=i/29, x=0.08+0.42*t, z=0.68+0.22*Math.sin(Math.PI*t*0.7+phase);
        hands.left.push([x,0.23+0.10*t,z]);
        hands.right.push([x,-0.23-0.10*t,z+0.04]);
    }
    return {type:"trajectory",version:1,id:"local-mock",frame:"robot_base",
            units:"m",hands:hands};
}
function connect() {
    if (!script.internetModule) { return; }
    try {
        socket=script.internetModule.createWebSocket(script.websocketUrl);
        socket.onopen=function(){lastNetworkError="";print("R1 AR: WebSocket connected to "+script.websocketUrl);};
        socket.onmessage=function(event){
            try {
                var trajectory = JSON.parse(event.data);
                applyTrajectory(trajectory);
                lastReceivedAt=getTime();
                if (!liveLogged && trajectory && trajectory.hands &&
                    isPath(trajectory.hands.left) && isPath(trajectory.hands.right)) {
                    print("R1 AR: live trajectory received ("+
                          trajectory.hands.left.length+" points per hand)");
                    liveLogged=true;
                }
            } catch(e) { print("R1 AR: invalid message: "+e); }
        };
        socket.onerror=function(event){
            if (lastNetworkError!=="connection") {
                print("R1 AR: WebSocket connection failed at "+script.websocketUrl+
                      "; using local mock ("+event+")");
                lastNetworkError="connection";
            }
        };
        socket.onclose=function(event){
            socket=null;reconnectAt=getTime()+2;
            if (event && event.code && event.code!==1000 && !lastNetworkError) {
                print("R1 AR: WebSocket closed with code "+event.code);
                lastNetworkError="closed";
            }
        };
    } catch(e) {
        if (lastNetworkError!=="unavailable") {
            print("R1 AR: WebSocket unavailable: "+e);
            lastNetworkError="unavailable";
        }
        socket=null;
        reconnectAt=String(e).indexOf("simulated platform")>=0 ? Infinity : getTime()+2;
    }
}
script.createEvent("UpdateEvent").bind(function(){
    updateAnchor();
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
    if (!socket && getTime()>=reconnectAt) { connect(); }
    if (getTime()-lastReceivedAt>1.5 && getTime()-lastMockAt>0.3) {
        applyTrajectory(localMock(getTime()*0.3));
        lastMockAt=getTime();
    }
});
