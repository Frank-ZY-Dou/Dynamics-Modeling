<h1 align="center">Hand_Robot_Control</h1>

<p align="center">Communication drivers, joint control, and examples for robotic hands.</p>

<p align="center">
  <a href="wuji2/">
    <img src="wuji2/docs/media/wuji2_gestures.gif" alt="Wuji Hand 2 gesture examples" width="640">
  </a>
</p>

<p align="center"><a href="wuji2/">Wuji Hand 2: connection, control, and six gestures</a></p>

Part of [Dynamics-Modeling](https://github.com/Frank-ZY-Dou/Dynamics-Modeling).

| Hand | Driver | Examples |
| --- | --- | --- |
| Wuji Hand 2 Beta 2 | [wuji2](wuji2/) | Open palm, pointing, peace, thumbs up, I love you, shaka |

Each hand has its own package, dependency declarations, tests, and device notes.
The Wuji2 package separates SDK communication, motion execution, model checks,
and recording. New gestures use the same controller and stop handling.

## Contributing

Keep hardware-specific communication inside its driver. Document the joint order,
units, supported hardware and SDK versions, and how commands stop when feedback
is lost. Tests should run without a connected hand; hardware observations belong
in a separate test report with the configuration used.

For Wuji2 development and tests, see [the package README](wuji2/README.md) and
[control design](wuji2/docs/control.md). The package is available under the
[MIT License](wuji2/LICENSE); upstream SDK and model licenses remain with their
respective projects.
