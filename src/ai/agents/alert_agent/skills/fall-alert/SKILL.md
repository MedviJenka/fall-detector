---
name: fall-alert
description: Conservative policy for classifying ordered fall-detection camera frames.
---

# Fall-event review policy

- Treat the four images as one chronological sequence, not independent scenes.
- Classify `fall` only when one visible person moves from upright or seated to
  the floor and remains down in the final frame.
- Classify deliberate sitting, lying down, exercise, and camera motion as
  `not_fall`.
- Classify `uncertain` when the person, transition, floor contact, or final
  posture is obscured.
- Set `person_visible` to true only when a person can actually be identified in
  the sequence.
- Base confidence only on visible evidence. Never infer injury, consciousness,
  intent, identity, age, or medical condition.
- Keep the reason factual and under 240 characters.