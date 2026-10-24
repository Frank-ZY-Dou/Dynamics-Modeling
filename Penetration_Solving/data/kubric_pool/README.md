# Kubric Objects

The data files for each object have been downloaded from Kubric's public assets:
`https://storage.googleapis.com/kubric-public/assets/GSO/<object_name>.tar.gz` for the 39
Google Scanned Objects, and
`https://storage.googleapis.com/kubric-public/assets/KuBasic/sphere.tar.gz` for `sphere`.

These Kubric assets are licensed CC BY-SA 4.0 (the `license` field in each `data.json`).
The original Google Scanned Objects scans are CC BY 4.0.

The physical simulations are computed with `collision_geometry.obj` files, which are
simplified / coarser meshes compared to the `visual_geometry.obj`, used to generate the
MP4 videos of each dataset sample.
