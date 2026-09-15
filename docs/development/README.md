# Development documentation

Current development references:

- [Project status](../project_status.md)
- [Architecture](../architecture.md)
- [Scene manifests](../scene-manifests.md)
- [Renderer conventions](../p0_renderer.md)
- [Camera conventions](../p1_camera.md)
- [Navigation controls](../navigation-controls.md)
- `scripts/README.md` for runtime, smoke, and utility script boundaries

Run the CPU regression suite from the repository root with:

```powershell
python -m pytest
```

Pytest is intentionally scoped to `tests/`; GPU-dependent scripts under
`scripts/smoke/` are explicit real-scene checks, not import-safe unit tests.

Historical phase plans and MVP reports are preserved in `docs/archive/` and
must not be treated as current behavior. Put new design decisions and failure
analyses here or in the maintained architecture/status documents.
