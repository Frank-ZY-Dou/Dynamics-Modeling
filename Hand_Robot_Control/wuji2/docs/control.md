# Communication and control

## Joint layout and units

Every command has 20 entries, ordered thumb, index, middle, ring, pinky, with
four joints per finger. For finger index `f` and local joint `j`, the SDK node
identifier is `5*f + j + 1`; the gaps between fingers are intentional.
State and diagnostic frames are independently checked for all 20 unique nodes.

Positions are joint-side radians; velocities are rad/s. The SDK calls the
current field `effort`, but its unit is amperes, not N·m. The position command
sets velocity and feedforward current to zero, leaving the device's PD loop to
track each successive position target.

Thumb joints are CMC flexion, CMC lateral swing, MCP and IP. The other fingers
use MCP flexion, MCP lateral swing, PIP and DIP. Model validation identifies
these joints by name and maps each angle to its model address.

## Connection ownership

A connection selects one serial number or one endpoint and checks handedness.
The SDK is loaded only when needed. Each driver owns its subscription handles,
publisher, and uniquely named manager connection. Closing one driver does not
disconnect other devices managed by the same process. Read-only inspection never
enables motors or writes control settings.

## Session lifecycle

1. Connect and obtain fresh, complete feedback. Require Ready state.
2. Save current gains and current caps, and preflight the requested paths from
   the measured initial pose.
3. Apply temporary settings and verify their readback. Recheck that the hand has
   not moved, repeat the initial-path preflight, and seed the first command with
   fresh measured angles.
4. Enable, confirm the state transition, and execute smooth position trajectories.
5. On completion or failure, stop commands and request disable. Confirm with fresh
   raw diagnostics, including when those diagnostics also carry a fault.
6. Restore and verify the saved settings only when disable is confirmed. Close
   all owned resources, retaining cleanup errors in the session result.

A partial enable or a partially completed settings write still requires cleanup.
An unsuccessful stop must not be represented as a disabled hand. The controller
never clears faults, recalibrates origins, updates firmware, or edits network settings.

## Trajectories and monitoring

Position interpolation uses `10u^3 - 15u^4 + 6u^5`, with zero endpoint velocity
and acceleration. Its peak normalized derivative is 1.875, which sets the minimum
move duration for the requested joint-speed limit. Commands are incremental and
are not sent in a catch-up burst after a scheduler delay.

The controller monitors complete state and diagnostic streams, current,
temperature, voltage, position and velocity limit flags, unexpected motor state,
tracking error, and measured joint speed. A separate watchdog observes the
control heartbeat. Stop requests are latched for the session; faults do not
automatically re-enable the hand.

The gesture CLI supplies recording health as an additional session guard. Camera
freshness and writer health are both required. A separate capture process owns
the OpenCV objects, allowing a blocked camera read to be terminated during
shutdown without releasing a capture concurrently from another thread.

## Session settings

[`ControlConfig`](../src/wuji2_control/config.py) validates all values as finite
numbers within the following inclusive ranges. The defaults apply to both the
CLI and Python API; the CLI exposes gains, current limit, commanded speed, and
temperature cutoff. Other settings are configurable through the Python API.

| Setting | Default | Allowed range | Meaning |
| --- | --- | --- | --- |
| `kp` | 3 | 3–5 | Device proportional gain |
| `kd` | 0.05 | 0.01–0.05 | Device derivative gain |
| `current_limit_A` | 0.5 A | 0.01–1.5 A | Temporary per-joint current cap |
| `speed_rad_s` | 0.2 rad/s | 0.01–0.3 rad/s | Peak commanded joint speed |
| `control_hz` | 100 Hz | 20–200 Hz | Maximum command loop frequency |
| `max_temperature_C` | 60 °C | 1–65 °C | MCU temperature cutoff |
| `feedback_timeout_s` | 0.3 s | 0.01–0.3 s | Maximum age of either feedback stream |
| `tracking_error_rad` | 0.25 rad | 0.01–0.25 rad | Allowed command-to-measurement error |
| `tracking_error_duration_s` | 0.5 s | 0.01–0.5 s | Time allowed above that error |
| `peak_speed_rad_s` | 4 rad/s | 0.1–4 rad/s | Measured-speed cutoff |
| `watchdog_timeout_s` | 0.75 s | 0.1–1.5 s | Controller heartbeat timeout |
| `max_hold_s` | 8 s | 0–30 s | Longest individual waypoint hold |
| `max_run_s` | 180 s | 1–300 s | Planned and enabled-run duration bound |
| `disable_timeout_s` | 2 s | 0.1–5 s | Wait for initial feedback or a state transition |
| `min_move_s` | 2 s | 0.1–5 s | Minimum interpolation duration |

The watchdog timeout must also exceed two control periods. Measured current
above the configured cap plus 0.3 A stops the session. Supply voltage must stay
strictly between 10.5 and 13.5 V. These checks supplement device protections.
The watchdog issues a best-effort stop request; device or transport failure can
prevent confirmation that motors are disabled.

## Model checks

The model must contain the expected 20 hinge joints for the selected hand.
Paths are sampled with the configured joint-limit margin and penetration
tolerance. This check catches modeled self-collisions; it does not contain the
physical workspace or identify cable and contact forces. There is no target
clipping that could introduce an uncommanded step at a limit.

## Tests

Device tests use a fake SDK to inject incomplete and invalid feedback and
failures at connection, settings, enable, disable, and restoration. Motion and
model tests run without hardware. Recording tests use fake capture/writer
objects and check the timestamp-to-frame mapping. CLI planning and import tests
must remain usable without a device, a camera, or the hardware SDK installed.
