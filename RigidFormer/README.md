# RigidFormer: Learning Rigid Dynamics using Transformers

**Project page:** [https://people.csail.mit.edu/frankzydou/projects/RigidFormer/index.html](https://people.csail.mit.edu/frankzydou/projects/RigidFormer/index.html)

Zhiyang Dou<sup>1</sup>, Minghao Guo<sup>1</sup>, Haixu Wu<sup>1</sup>, Doug Roble<sup>2</sup>, Tuur Stuyck<sup>2</sup>, Wojciech Matusik<sup>1</sup>

<sup>1</sup> Massachusetts Institute of Technology &nbsp;&nbsp;&nbsp; <sup>2</sup> Meta

---

## Abstract

Learning-based simulation of multi-object rigid-body dynamics remains difficult because contact is discontinuous and errors compound over long horizons. Most existing methods remain tied to mesh connectivity and vertex-level message passing, which limits their applicability to mesh-free inputs such as point clouds and leads to high computational cost. Efficiently modeling high-fidelity rigid-body dynamics from mesh-free representations therefore remains challenging.

We introduce **RigidFormer**, an object-centric Transformer-based model that learns mesh-free rigid-body dynamics with controllable integration step sizes. RigidFormer reasons at the *object level* and advances each object through compact anchors; *Anchor-Vertex Pooling* enriches these anchors with local vertex features, retaining contact-relevant geometry without dense vertex-level interaction. We propose *Anchor-based RoPE* to inject anchor geometry into attention while respecting the unordered nature of objects and anchors: object-token processing is permutation-equivariant, and the mean-pooled anchor descriptor is invariant to anchor reindexing while preserving shape extent. RigidFormer further enforces *rigidity* by projecting updates onto the rigid-body manifold using differentiable Kabsch alignment.

On standard benchmarks, RigidFormer outperforms or matches mesh-based baselines using point inputs, runs faster, generalizes to unseen point resolutions and across datasets, and scales to 200+ objects; we also show a preliminary extension to command-conditioned articulated bodies by treating body parts as interacting object-level components.

---

## Code

Code release is coming soon. Please refer to the [project page](https://people.csail.mit.edu/frankzydou/projects/RigidFormer/index.html) for updates.
