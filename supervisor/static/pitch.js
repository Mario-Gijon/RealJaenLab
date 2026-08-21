const PITCH_MODEL = {
  length: 105,
  width: 68,
  centerCircleRadius: 9.15,
  landmarks: [
    ["top_left_corner", "Top-left pitch corner", -52.5, -34],
    ["top_right_corner", "Top-right pitch corner", 52.5, -34],
    ["bottom_right_corner", "Bottom-right pitch corner", 52.5, 34],
    ["bottom_left_corner", "Bottom-left pitch corner", -52.5, 34],
    ["halfway_top", "Halfway line / top touchline", 0, -34],
    ["halfway_bottom", "Halfway line / bottom touchline", 0, 34],
    ["centre_mark", "Centre mark", 0, 0],
    ["left_penalty_top", "Left penalty-area top corner", -36, -20.16],
    ["left_penalty_bottom", "Left penalty-area bottom corner", -36, 20.16],
    ["right_penalty_top", "Right penalty-area top corner", 36, -20.16],
    ["right_penalty_bottom", "Right penalty-area bottom corner", 36, 20.16],
    ["left_goal_top", "Left goal-area top corner", -47, -9.16],
    ["left_goal_bottom", "Left goal-area bottom corner", -47, 9.16],
    ["right_goal_top", "Right goal-area top corner", 47, -9.16],
    ["right_goal_bottom", "Right goal-area bottom corner", 47, 9.16],
    ["left_penalty_spot", "Left penalty spot", -41.5, 0],
    ["right_penalty_spot", "Right penalty spot", 41.5, 0],
  ].map(([id, label, x, y]) => ({ id, label, world: { x, y } })),
};

PITCH_MODEL.landmarksById = Object.fromEntries(PITCH_MODEL.landmarks.map((landmark) => [landmark.id, landmark]));

PITCH_MODEL.segments = [
  [{ x: -52.5, y: -34 }, { x: 52.5, y: -34 }, { x: 52.5, y: 34 }, { x: -52.5, y: 34 }, { x: -52.5, y: -34 }],
  [{ x: 0, y: -34 }, { x: 0, y: 34 }],
  [{ x: -52.5, y: -20.16 }, { x: -36, y: -20.16 }, { x: -36, y: 20.16 }, { x: -52.5, y: 20.16 }],
  [{ x: 52.5, y: -20.16 }, { x: 36, y: -20.16 }, { x: 36, y: 20.16 }, { x: 52.5, y: 20.16 }],
  [{ x: -52.5, y: -9.16 }, { x: -47, y: -9.16 }, { x: -47, y: 9.16 }, { x: -52.5, y: 9.16 }],
  [{ x: 52.5, y: -9.16 }, { x: 47, y: -9.16 }, { x: 47, y: 9.16 }, { x: 52.5, y: 9.16 }],
];

PITCH_MODEL.sampleCircle = (center, radius, samples = 96) => (
  Array.from({ length: samples + 1 }, (_, index) => {
    const angle = (index / samples) * Math.PI * 2;
    return { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) };
  })
);
