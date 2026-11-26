# Scene backgrounds

Keep each background's textures and related assets in its own theme folder:

```text
assets/background/
└── christmas/
```

Add future backgrounds as sibling folders. Reusable scene helpers should
resolve this root through `utils.paths.BACKGROUND_ASSETS_DIR`.
